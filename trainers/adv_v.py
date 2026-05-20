import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from copy import deepcopy
import torch.backends.cudnn as cudnn
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import math
import json
import random
from torch.utils.data import Dataset
import numpy as np
from utils import Summary, ProgressMeter, accuracy, load_model_weight, set_random_seed, AverageMeter
import time
from tqdm import tqdm

from clip import clip
from clip import tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

from fvcore.nn import FlopCountAnalysis

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'VPT',
                      "vision_depth": cfg.TRAINER.TAPTV.PROMPT_DEPTH_VISION,
                      "vision_ctx": cfg.TRAINER.TAPTV.N_CTX_VISION,
                      "language_depth": 0,
                      "language_ctx": 0}
    assert cfg.TRAINER.TAPTV.PROMPT_DEPTH_VISION >= 1, "For Vision Prompting, PROMPT_DEPTH_VISION should be >= 1"
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class FixedEmbeddings():
    def __init__(self, cfg, classnames, clip_model):
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        prompt_prefix = "a photo of a"
        print('Adversarial Vision Prompting Design')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Adversarial Vision prompting: {cfg.TRAINER.TAPTV.N_CTX_VISION}")
        print(f"Using fixed hand crated prompts")

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            text_features = clip_model.encode_text(tokenized_prompts)

        self.fixed_embeddings = text_features

    def return_fixed_embeddings(self):
        return self.fixed_embeddings

    def reset(self):

        pass

    def reset_classnames(self, classnames, args):

        pass

    def set_prompt_init_states(self):

        pass

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.embeddings = FixedEmbeddings(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.vpt_initial_states = {}
        for name, param in self.named_parameters():
            if "VPT" in name:
                self.vpt_initial_states[name] = param.detach().clone()

        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    def forward(self, image, label=None, training=False):
        image = self.normalize(image)

        logit_scale = self.logit_scale.exp()

        text_features = self.embeddings.return_fixed_embeddings().cuda()
        image_features = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        if training:
            return F.cross_entropy(logits, label)

        return logits

    def get_text_features(self):
        with torch.no_grad():
            text_features = self.embeddings.return_fixed_embeddings().cuda()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        # self.prompt_learner.reset()
        for name, param in self.named_parameters():
            if "VPT" in name:
                # print(name)
                param.copy_(self.vpt_initial_states[name])
                
    def reset_classnames(self, classnames, arch):
        # self.prompt_learner.reset_classnames(classnames, arch)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        pass

    def set_prompt_inits(self):
        # print("Re-updating prompt initializations to current prompts.")
        # self.prompt_learner.set_prompt_init_states()
        pass

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


ID_to_DIRNAME={
    'PUG': 'PUG_ImageNet',
    'I': 'imagenet/images',
    'A': 'imagenet-adversarial/imagenet-a',
    'K': 'imagenet-sketch/images',
    'R': 'imagenet-rendition/imagenet-r',
    'V': 'imagenetv2/imagenetv2-matched-frequency-format-val',
    'flower102': 'oxford_flowers',
    'dtd': 'dtd',
    'pets': 'oxford_pets',
    'cars': 'stanford_cars',
    'ucf101': 'ucf101',
    'caltech101': 'caltech-101',
    'food101': 'food-101',
    'sun397': 'sun397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat'
}


class BaseJsonDataset(Dataset):
    def __init__(self, image_path, json_path, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.image_path = image_path
        self.split_json = json_path
        self.mode = mode
        self.image_list = []
        self.label_list = []

        with open(self.split_json) as fp:
            splits = json.load(fp)
            samples = splits[self.mode]
            for s in samples:
                self.image_list.append(s[0])
                self.label_list.append(s[1])
    
        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_path, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()

fewshot_datasets = ['dtd', 'flower102', 'food101', 'cars', 'sun397', 
                    'aircraft', 'pets', 'caltech101', 'ucf101', 'eurosat']

path_dict = {
    # dataset_name: ["image_dir", "json_split_file"]
    "flower102": ["jpg", "split_zhou_OxfordFlowers.json"],
    "food101": ["images", "split_zhou_Food101.json"],
    "dtd": ["images", "split_zhou_DescribableTextures.json"],
    "pets": ["images", "split_zhou_OxfordPets.json"],
    "sun397": ["SUN397", "split_zhou_SUN397.json"],
    "caltech101": ["101_ObjectCategories", "split_zhou_Caltech101.json"],
    "ucf101": ["UCF-101-midframes", "split_zhou_UCF101.json"],
    "cars": ["", "split_zhou_StanfordCars.json"],
    "eurosat": ["2750", "split_zhou_EuroSAT.json"]
}

pug_setting_dir = {
    'CRoll': 'Camera_Roll',
    'CPitch': 'Camera_Pitch',
    'CYaw': 'Camera_Yaw',
    'OPitch': 'Object_Pitch',
    'ORoll': 'Object_Roll',
    'OScale': 'Object_Scale',
    'OTexture': 'Object_Texture',
    'OYaw': 'Object_Yaw',
    'SLight': 'Scene_Light',
    'Worlds': 'Worlds'
}

class Aircraft(Dataset):
    """ FGVC Aircraft dataset """
    def __init__(self, root, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.path = root
        self.mode = mode

        self.cname = []
        with open(os.path.join(self.path, "variants.txt"), 'r') as fp:
            self.cname = [l.replace("\n", "") for l in fp.readlines()]

        self.image_list = []
        self.label_list = []
        with open(os.path.join(self.path, 'images_variant_{:s}.txt'.format(self.mode)), 'r') as fp:
            lines = [s.replace("\n", "") for s in fp.readlines()]
            for l in lines:
                ls = l.split(" ")
                img = ls[0]
                label = " ".join(ls[1:])
                self.image_list.append("{}.jpg".format(img))
                self.label_list.append(self.cname.index(label))

        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, 'images', self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()


# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()   # Resizing with scaling and ratio
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views

class AdvAugMixAugmenter(object):
    def __init__(self, preprocess, n_views=2, augmix=False, severity=1):
        self.preprocess = preprocess
        self.n_views = n_views
        self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(x)
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views


class TAPTV(TrainerX):
    def build_pug_dataset(self, set_id, data_root, transform):
        setting = set_id.split('_')[1]
        pug_dir = pug_setting_dir[setting]
        testdir = os.path.join(data_root, ID_to_DIRNAME['PUG'], pug_dir)
        testset = datasets.ImageFolder(testdir, transform=transform)
        return testset

    def build_fewshot_dataset(self, set_id, root, transform, mode='train', n_shot=None):
        if set_id.lower() == 'aircraft':
            return Aircraft(root, mode, n_shot, transform)
        path_suffix, json_path = path_dict[set_id.lower()]
        json_path = os.path.join(root, json_path)
        image_path = os.path.join(root, path_suffix)
        return BaseJsonDataset(image_path, json_path, mode, n_shot, transform)

    def build_dataset(self, set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False):
        if set_id == 'I':
            # ImageNet validation set
            testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
            testset = datasets.ImageFolder(testdir, transform=transform)
        elif set_id in ['A', 'K', 'R', 'V']:
            testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
            testset = datasets.ImageFolder(testdir, transform=transform)
        elif set_id in fewshot_datasets:
            if mode == 'train' and n_shot:
                testset = self.build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
            else:
                testset = self.build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)
        elif 'PUG' in set_id:
            testset = self.build_pug_dataset(set_id, data_root, transform=transform)
        else:
            raise NotImplementedError
            
        return testset

    def build_data_loader(self):
        super().build_data_loader()
        print("Test-Time Adversarial Loader: ", self.cfg.AT.ADVALIGN)
        if self.cfg.AT.ADVALIGN:
            self.tapt_loader = self.get_tapt_adv_dataloader(self.cfg.TAPT)
        else:
            self.tapt_loader = self.get_tapt_dataloader(self.cfg.TAPT)

    def get_tapt_dataloader(self, args):
        print("Loading pre-computed means and vars")
        self.visual_vars = torch.load(args.VIS_VARS)
        self.visual_means = torch.load(args.VIS_MEANS)
        self.visual_vars_clean = torch.load(args.VIS_VARS_CLEAN)
        self.visual_means_clean = torch.load(args.VIS_MEANS_CLEAN)
        print("source adv visual vars: {}, Path: {}".format(self.visual_vars.shape, args.VIS_VARS))
        print("source adv visual means: {}, Path: {}".format(self.visual_means.shape, args.VIS_MEANS))
        print("source clean visual vars: {}, Path: {}".format(self.visual_vars_clean.shape, args.VIS_VARS_CLEAN))
        print("source clean visual means: {}, Path: {}".format(self.visual_means_clean.shape, args.VIS_MEANS_CLEAN))

        # normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
        #                                  std=[0.26862954, 0.26130258, 0.27577711])
        tapt = args.RUN
        if tapt:
            base_transform = transforms.Compose([
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                # normalize,
            ])
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.BATCH_SIZE-1, 
                                            augmix=False)
            batchsize = 1
        else:
            data_transform = transforms.Compose([
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # normalize,
            ])
            batchsize = args.BATCH_SIZE

        set_id = self.cfg.DATASET.TAPT
        val_dataset = self.build_dataset(set_id, data_transform, self.cfg.DATASET.ROOT, mode='test')
        # print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=True,
                    num_workers=8, pin_memory=True)
        
        return val_loader
    
    def test_time_adapt_eval(self, val_loader, model, optimizer, optim_state, scaler, args):
        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        if self.cfg.AT.ADVALIGN:
            top1 = AverageMeter('Robust Acc@1', ':6.2f', Summary.AVERAGE)
            top5 = AverageMeter('Robust Acc@5', ':6.2f', Summary.AVERAGE)
        else:
            top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
            top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, top5],
            prefix='Test: ')
        print("="*40)
        print(f"Running for {args.BATCH_SIZE} Augmented views")
        print(f"Running for {args.TTA_STEPS} TTA steps")

        # reset model and switch to evaluate mode
        model.eval()
        if not args.COCOOP: # no need to reset cocoop because it's fixed
            with torch.no_grad():
                model.reset()
        end = time.time()
        for i, batch in enumerate(val_loader):
            # images, target = self.parse_batch_test(batch)
            images, target = batch

            # assert args.gpu is not None
            if isinstance(images, list):
                for k in range(len(images)):
                    # images[k] = images[k].cuda(args.gpu, non_blocking=True)
                    images[k] = images[k].to(self.device)
                image = images[0]
            else:
                if len(images.size()) > 4:
                    # when using ImageNet Sampler as the dataset
                    assert images.size()[0] == 1
                    images = images.squeeze(0)
                # images = images.cuda(args.gpu, non_blocking=True)
                images = images.to(self.device)
                image = images
            # target = target.cuda(args.gpu, non_blocking=True)
            target = target.to(self.device)
            if args.RUN:
                images = torch.cat(images, dim=0)

            # reset the tunable prompt to its initial state
            if not args.COCOOP: # no need to reset cocoop because it's fixed
                if args.TTA_STEPS > 0:
                    with torch.no_grad():
                        model.reset()
                optimizer.load_state_dict(optim_state)
                self.test_time_tuning(model, images, optimizer, scaler, args)
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        image_feature, pgen_ctx = model.gen_ctx(images, args.RUN)
                optimizer = None
                pgen_ctx = self.test_time_tuning(model, (image_feature, pgen_ctx), optimizer, scaler, args)

            # The actual inference goes here
            if args.RUN:
                if args.COCOOP:
                    image_feature = image_feature[0].unsqueeze(0)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    if args.COCOOP:
                        output = model((image_feature, pgen_ctx))
                    else:
                        output = model(image)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % 200 == 0:
                progress.display(i)

        progress.display_summary()

        return [top1.avg, top5.avg]

    def test_time_tuning(self, model, inputs, optimizer, scaler, args):
        if args.COCOOP:
            image_feature, pgen_ctx = inputs
            pgen_ctx.requires_grad = True
            optimizer = torch.optim.AdamW([pgen_ctx], args.LR)
        
        selected_idx = None
        for j in range(args.TTA_STEPS):
            with torch.cuda.amp.autocast():
                if args.COCOOP:
                    output = model((image_feature, pgen_ctx))
                else:
                    output = model(inputs) 

                if selected_idx is not None:
                    output = output[selected_idx]
                else:
                    output, selected_idx = self.select_confident_samples(output, args.TAPT_THRESHOLD, args.ALIGN_THRESHOLD)

                if args.TAPT_LOSS:
                    loss = self.avg_entropy(output)

                # Only selected indexes
                target_feat_distr = (self.visual_means, self.visual_vars)
                out_visual_mean = torch.cat([torch.mean(res.visual_feat[:, selected_idx, :], dim=1, keepdims=True).permute(1,0,2) for res in model.image_encoder.transformer.resblocks])
                out_visual_var = torch.cat([torch.mean(((res.visual_feat[:, selected_idx, :] - out_visual_mean[i, :, :].unsqueeze(0).permute(1,0,2))**2), dim=1, keepdims=True).permute(1,0,2) for i, res in enumerate(model.image_encoder.transformer.resblocks)])
                out_feat_distr = (out_visual_mean, out_visual_var)

                if args.DISTR_ALIGN:
                    DISTR_LOSS_W = args.DISTR_LOSS_W / (args.ALIGN_LAYER_TO - args.ALIGN_LAYER_FROM)
                    if not args.TAPT_LOSS:
                        loss = DISTR_LOSS_W * self.distr_align_loss(out_feat_distr, target_feat_distr, 
                                                layers_from=args.ALIGN_LAYER_FROM, layers_to=args.ALIGN_LAYER_TO)
                    else: 
                        loss += DISTR_LOSS_W * self.distr_align_loss(out_feat_distr, target_feat_distr, 
                                                layers_from=args.ALIGN_LAYER_FROM, layers_to=args.ALIGN_LAYER_TO)
            
            optimizer.zero_grad()
            # compute gradient and do SGD step
            scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.step(optimizer)
            scaler.update()
        if args.COCOOP:
            return pgen_ctx

        return
    
    def select_confident_samples(self, logits, topTAPT, topAlign):
        batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        idxTAPT = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * topTAPT)]
        idxAlign = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * topAlign)]
        return logits[idxTAPT], idxAlign

    def avg_entropy(self, outputs):
        logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
        avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
        min_real = torch.finfo(avg_logits.dtype).min
        avg_logits = torch.clamp(avg_logits, min=min_real)
        return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)
    
    def distr_align_loss(self, out_feat, targ_feat, layers_from=0, layers_to=12, moments=5):
        '''
        A feature distibution alignment L1 loss between mean and variance of the features
        '''
        distr_loss = 0
        out_means, out_vars = out_feat
        targ_means, targ_vars = targ_feat
        transf_layers = layers_to
        for l in range(layers_from, transf_layers-1):
            out_mean, out_var = out_means[l], out_vars[l]
            targ_mean, targ_var = targ_means[l], targ_vars[l]
            distr_loss += 0.5 * F.l1_loss(out_mean, targ_mean) + 0.5 * F.l1_loss(out_var, targ_var)
        return distr_loss


    ################# TAPT CHANGES END #######################

    def check_cfg(self, cfg):
        assert cfg.TRAINER.TAPTV.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.TAPTV.PREC == "fp32" or cfg.TRAINER.TAPTV.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.TAPTV.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.TAPTV.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


@TRAINER_REGISTRY.register()
class TAPTV(TAPTV):
    def before_adv_align(self):
        r"""
        Arguments:
            eps (float): maximum perturbation. (Default: 4/255)
            alpha (float): step size. (Default: 1/255)
            steps (int): number of steps. (Default: 10)
            random_start (bool): using random initialization of delta. (Default: True)
        """
        
        adv_dataset_pkl_path = os.path.join(self.cfg.AT.ADV_DIR, self.cfg.DATASET.NAME + "_adv_dataset.pkl")
        print("load adv dataset: {}".format(adv_dataset_pkl_path))

        # If inputs were normalized, then
        # attacker.set_normalization_used(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

        # save_path = "/cpfs01/projects-HDD/cfff-12667fb5c6f8_HDD/wx_22110240041/wangxin/Test-Prompt/multimodal-prompt-learning/output/evaluation/AdvMaPLe/vit_b16_c2_ep100_batch32_2ctx_9depth_cross_datasets_lr0.0035_16shots/oxford_pets/seed1/100/OxfordPets_adv.pkl"
        if os.path.isfile(adv_dataset_pkl_path):
            self.adv_test_pkl, _ = torch.load(adv_dataset_pkl_path, weights_only=False).tensors
            return
        else:
            raise


    def get_tapt_adv_dataloader(self, args):
        print("Loading pre-computed means and vars")
        self.visual_vars = torch.load(args.VIS_VARS)
        self.visual_means = torch.load(args.VIS_MEANS)
        self.visual_vars_clean = torch.load(args.VIS_VARS_CLEAN)
        self.visual_means_clean = torch.load(args.VIS_MEANS_CLEAN)
        print("source adv visual vars: {}, Path: {}".format(self.visual_vars.shape, args.VIS_VARS))
        print("source adv visual means: {}, Path: {}".format(self.visual_means.shape, args.VIS_MEANS))
        print("source clean visual vars: {}, Path: {}".format(self.visual_vars_clean.shape, args.VIS_VARS_CLEAN))
        print("source clean visual means: {}, Path: {}".format(self.visual_means_clean.shape, args.VIS_MEANS_CLEAN))

        # normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
        #                                  std=[0.26862954, 0.26130258, 0.27577711])
        tapt = args.RUN
        if tapt:
            print("tapt: ", tapt)
            preprocess = transforms.Compose([
                # Add any required transformations
                # transforms.ToTensor(),
                # normalize,
                ])
            data_transform = AdvAugMixAugmenter(preprocess, n_views=args.BATCH_SIZE-1, augmix=False)
            batchsize = 1
        else:
            data_transform = transforms.Compose([
                # Add any required transformations
            ])
            batchsize = args.BATCH_SIZE

        # Load the adversarial dataset
        # print(self.cfg.AT.ADV_DIR)
        adv_dataset_pkl_path = os.path.join(self.cfg.AT.ADV_DIR, self.cfg.DATASET.NAME + "_adv_dataset.pkl")
        adv_dataset_pkl = torch.load(adv_dataset_pkl_path, weights_only=False)  # Assuming the .pkl file contains a dataset object
        print("load adv dataset: {}".format(adv_dataset_pkl_path))

        # Wrap the loaded dataset with TransformDataset to apply transformations
        val_dataset = TransformDataset(adv_dataset_pkl, transform=data_transform)

        # print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=True,
                    num_workers=8, pin_memory=True)
        
        return val_loader

    def tapt(self):
        """
        Run Test-time prompt Tuning
        """
        self.model.set_prompt_inits()   # Init with current prompts
        for name, param in self.model.named_parameters():
            if not self.cfg.TAPT.COCOOP: # MaPLe and CoOp
                if "VPT" not in name:
                    param.requires_grad_(False)
            else:
                if "text_encoder" not in name:
                    param.requires_grad_(False)
            
        # define optimizer
        if self.cfg.TAPT.COCOOP:
            optimizer = None
            optim_state = None
        else:
            # trainable_param = self.model.prompt_learner.parameters()
            trainable_param = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name)
                    trainable_param.append(param)            
            optimizer = torch.optim.AdamW(trainable_param, self.cfg.TAPT.LR)
            optim_state = deepcopy(optimizer.state_dict())

        # setup automatic mixed-precision (Amp) loss scaling
        scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        # print(self.model)
        # for name, parameters in self.model.named_parameters():
        #     print(name + " " + str(parameters.requires_grad))

        print('=> Using native Torch AMP. Training in mixed precision.')
        print("number of test samples: {}".format(len(self.tapt_loader.dataset)))
        print("DATASET.TAPT: {}".format(self.cfg.DATASET.TAPT))
        cudnn.benchmark = True

        results = {}
        set_id = self.cfg.DATASET.TAPT
        results[set_id] = self.test_time_adv_adapt_eval(self.tapt_loader, self.model, optimizer, optim_state, scaler, self.cfg.TAPT)
        return results

    def test_time_adv_adapt_eval(self, val_loader, model, optimizer, optim_state, scaler, args):
        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        if self.cfg.AT.ADVALIGN:
            top1 = AverageMeter('Robust Acc@1', ':6.2f', Summary.AVERAGE)
            top5 = AverageMeter('Robust Acc@5', ':6.2f', Summary.AVERAGE)
        else:
            top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
            top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, top5],
            prefix='Test: ')
        print("="*40)
        print(f"Running for {args.BATCH_SIZE} Augmented views")
        print(f"Running for {args.TTA_STEPS} TTA steps")

        # reset model and switch to evaluate mode
        model.eval()
        if not args.COCOOP: # no need to reset cocoop because it's fixed
            with torch.no_grad():
                model.reset()
        end = time.time()
        for i, batch in enumerate(val_loader):
            # images, target = self.parse_batch_test(batch)
            images, target = batch

            # assert args.gpu is not None
            if isinstance(images, list):
                for k in range(len(images)):
                    # images[k] = images[k].cuda(args.gpu, non_blocking=True)
                    images[k] = images[k].to(self.device)
                image = images[0]
            else:
                if len(images.size()) > 4:
                    # when using ImageNet Sampler as the dataset
                    assert images.size()[0] == 1
                    images = images.squeeze(0)
                # images = images.cuda(args.gpu, non_blocking=True)
                images = images.to(self.device)
                image = images
            # target = target.cuda(args.gpu, non_blocking=True)
            target = target.to(self.device)
            if args.RUN:
                images = torch.cat(images, dim=0)

            # reset the tunable prompt to its initial state
            if not args.COCOOP: # no need to reset cocoop because it's fixed
                if args.TTA_STEPS > 0:
                    with torch.no_grad():
                        model.reset()
                optimizer.load_state_dict(optim_state)
                self.test_time_adv_tuning(model, images, optimizer, scaler, args)
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        image_feature, pgen_ctx = model.gen_ctx(images, args.RUN)
                optimizer = None
                pgen_ctx = self.test_time_adv_tuning(model, (image_feature, pgen_ctx), optimizer, scaler, args)

            # The actual inference goes here
            if args.RUN:
                if args.COCOOP:
                    image_feature = image_feature[0].unsqueeze(0)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    if args.COCOOP:
                        output = model((image_feature, pgen_ctx))
                    else:
                        output = model(image)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % 200 == 0:
                progress.display(i)

        progress.display_summary()

        return [top1.avg, top5.avg]

    def test_time_adv_tuning(self, model, inputs, optimizer, scaler, args):
        if args.COCOOP:
            image_feature, pgen_ctx = inputs
            pgen_ctx.requires_grad = True
            optimizer = torch.optim.AdamW([pgen_ctx], args.LR)
        
        selected_idx = None
        gamma = 0.5 # 1 adv, 0 clea

        for j in range(args.TTA_STEPS):
            with torch.cuda.amp.autocast():
                if args.COCOOP:
                    output = model((image_feature, pgen_ctx))
                else:
                    output = model(inputs) 

                if selected_idx is not None:
                    output = output[selected_idx]
                else:
                    output, selected_idx = self.select_confident_samples(output, args.TAPT_THRESHOLD, args.ALIGN_THRESHOLD)

                if args.TAPT_LOSS:
                    loss = self.avg_entropy(output)

                # Only selected indexes
                target_feat_distr = (self.visual_means, self.visual_vars)
                target_clean_feat_distr = (self.visual_means_clean, self.visual_vars_clean)
                out_visual_mean = torch.cat([torch.mean(res.visual_feat[:, selected_idx, :], dim=1, keepdims=True).permute(1,0,2) for res in model.image_encoder.transformer.resblocks])
                out_visual_var = torch.cat([torch.mean(((res.visual_feat[:, selected_idx, :] - out_visual_mean[i, :, :].unsqueeze(0).permute(1,0,2))**2), dim=1, keepdims=True).permute(1,0,2) for i, res in enumerate(model.image_encoder.transformer.resblocks)])
                out_feat_distr = (out_visual_mean, out_visual_var)

                if args.DISTR_ALIGN:
                    DISTR_LOSS_W = args.DISTR_LOSS_W / (args.ALIGN_LAYER_TO - args.ALIGN_LAYER_FROM)

                    # Calculate the distribution alignment loss components
                    if self.cfg.AT.ALIGN_TYPE == "multi":
                        align_loss_func = lambda x, y: self.multistage_align_loss(x, y, current_step=j, args=args, switch_fraction=0.5)
                    else:
                        align_loss_func = lambda x, y: self.onestage_align_loss(x, y, layers_from=args.ALIGN_LAYER_FROM, layers_to=args.ALIGN_LAYER_TO, align_type=self.cfg.AT.ALIGN_TYPE)

                    loss_tapt_adv = align_loss_func(out_feat_distr, target_feat_distr)
                    loss_tapt_clean = align_loss_func(out_feat_distr, target_clean_feat_distr)

                    distr_loss = DISTR_LOSS_W * (loss_tapt_adv * gamma + loss_tapt_clean * (1 - gamma))

                    loss = distr_loss if not args.TAPT_LOSS else loss + distr_loss

                # if args.DISTR_ALIGN:
                #     DISTR_LOSS_W = args.DISTR_LOSS_W / (args.ALIGN_LAYER_TO - args.ALIGN_LAYER_FROM)
                #     if not args.TAPT_LOSS:
                #         loss = DISTR_LOSS_W * self.distr_align_loss(out_feat_distr, target_feat_distr, 
                #                                 layers_from=args.ALIGN_LAYER_FROM, layers_to=args.ALIGN_LAYER_TO)
                #     else: 
                #         loss += DISTR_LOSS_W * self.distr_align_loss(out_feat_distr, target_feat_distr, 
                #                                 layers_from=args.ALIGN_LAYER_FROM, layers_to=args.ALIGN_LAYER_TO)

            optimizer.zero_grad()
            # compute gradient and do SGD step
            scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.step(optimizer)
            scaler.update()
        if args.COCOOP:
            return pgen_ctx

        return

    def multistage_align_loss(self, out_feat_distr, target_feat_distr, current_step, args, switch_fraction=0.5):
        """
        Calculates the multi-stage distribution alignment loss.

        Args:
            out_feat_distr: Tuple of (out_visual_mean, out_visual_var) for the current batch.
            target_feat_distr: Tuple of (target_visual_mean, target_visual_var) for the target distribution.
            current_step: Current step in the align process.
            args: Align arguments.

        Returns:
            The calculated multi-stage loss.
        """
        # Sigmoid smooth interpolation switch
        k = 2  # Controls the steepness of the sigmoid function
        alpha = 1 / (1 + math.exp(-k * (current_step - args.TTA_STEPS * switch_fraction)))  # Calculate the alpha value using sigmoid

        # Define helper function for calculating loss
        def calculate_loss(align_type):
            return self.onestage_align_loss(out_feat_distr, target_feat_distr, 
                                            layers_from=args.ALIGN_LAYER_FROM,
                                            layers_to=args.ALIGN_LAYER_TO, 
                                            align_type=align_type)

        # Calculate L1 and MMD losses
        loss_l1 = calculate_loss('l1')
        loss_mmd = calculate_loss('mmd')

        return (1 - alpha) * loss_mmd + alpha * loss_l1

    def onestage_align_loss(self, out_feat, targ_feat, layers_from=0, layers_to=12, align_type='l1', sigma=1.0):
        '''
        A feature distribution alignment loss between mean and variance of the features.
        '''
        distr_loss = 0
        out_means, out_vars = out_feat
        targ_means, targ_vars = targ_feat

        # Choose the correct loss function based on align_type
        for l in range(layers_from, layers_to - 1):
            out_mean, out_var = out_means[l], out_vars[l]
            targ_mean, targ_var = targ_means[l], targ_vars[l]

            # Compute the appropriate loss based on align_type
            if align_type == 'l1':
                distr_loss += 0.5 * F.l1_loss(out_mean, targ_mean) + 0.5 * F.l1_loss(out_var, targ_var)
            elif align_type == 'l2':
                distr_loss += 0.5 * F.mse_loss(out_mean, targ_mean) + 0.5 * F.mse_loss(out_var, targ_var)
            elif align_type == 'mmd':
                distr_loss += self.compute_mmd(out_mean, targ_mean, sigma)
            elif align_type == 'kl':
                distr_loss += F.kl_div(F.log_softmax(out_mean, dim=-1), F.softmax(targ_mean, dim=-1), reduction='batchmean') 
            elif align_type == 'js':
                distr_loss += self.compute_js_div(out_mean, targ_mean)
            # elif align_type == 'wasserstein':
            #     distr_loss += torch.mean(torch.abs(out_mean - targ_mean)) + torch.mean(torch.abs(torch.sqrt(out_var) - torch.sqrt(targ_var)))
            # elif align_type == 'cosine':
            #     distr_loss += 1 - F.cosine_similarity(out_mean, targ_mean, dim=-1).mean()
            else:
                raise ValueError(f"Invalid align_type: {align_type}")

        return distr_loss

    def compute_mmd(self, x, y, sigma=1.0):
        """
        Compute the Maximum Mean Discrepancy (MMD) between two samples.
        """
        x_kernel = self.gaussian_kernel(x, x, sigma)
        y_kernel = self.gaussian_kernel(y, y, sigma)
        xy_kernel = self.gaussian_kernel(x, y, sigma)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

    def gaussian_kernel(self, x, y, sigma):
        """
        Compute the Gaussian kernel between two samples.
        """
        return torch.exp(-torch.sum((x[:, None, :] - y[None, :, :]) ** 2, dim=2) / (2 * (sigma ** 2)))

    def compute_js_div(self, p, q):
        """
        Compute the Jensen-Shannon Divergence between two probability distributions.
        """
        m = 0.5 * (p + q)
        return 0.5 * (F.kl_div(F.log_softmax(p, dim=-1), F.softmax(m, dim=-1), reduction='batchmean') + F.kl_div(F.log_softmax(q, dim=-1), F.softmax(m, dim=-1), reduction='batchmean'))

    @torch.no_grad()
    def save_and_compute_feature_maps(self, split=None, save_path='./output/TTAPT/features/'):
        """
        Saving feature maps (i.e. tokens from transformer)
        """
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        elif split == "test":   # in case val_loader is None
            data_loader = self.test_loader
        elif split == "train":  # in case val_loader is None
            data_loader = self.train_loader_x

        print(f"Calculate var and mean on the *{split}* set")        
        all_visual_feats = []

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)

            visual_feats = torch.stack([res.visual_feat.permute(1, 0, 2) for res in self.model.image_encoder.transformer.resblocks], dim=1)
            all_visual_feats.append(visual_feats.cpu())

            # Process output and labels
            self.evaluator.process(output, label)

        # Concatenate all the visual features across batches and compute the mean
        all_visual_feats = torch.cat(all_visual_feats, dim=0)
        print("all_visual_feats shape: ", all_visual_feats.shape)

        # mean_visual_feat = all_visual_feats.mean(dim=0).cuda()
        var_visual_feat = all_visual_feats.var(dim=0).cuda()

        del all_visual_feats

        print(f"******Saving feature maps to {save_path}*********")
        # torch.save(mean_visual_feat, save_path + "ImgNetpre_vis_means_{}_half_eps4_ep100_{}shot.pt".format(split, self.cfg.DATASET.NUM_SHOTS))
        torch.save(var_visual_feat, save_path + "ImgNetpre_vis_vars_{}_half_eps4_ep100_{}shot.pt".format(split, self.cfg.DATASET.NUM_SHOTS))

        results = self.evaluator.evaluate()

        # Save evaluation results
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    @torch.no_grad()
    def save_and_compute_means(self, split=None, save_path='./stats/vitb32/'):
        """
        Saving feature maps (i.e. tokens from transformer) and calculating mean in batches
        """
        self.set_model_mode("eval")
        self.evaluator.reset()
        
        # summary(self.model, input_size=(32, 3, 224, 224))
        
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        elif split == "test":   # in case val_loader is None
            data_loader = self.test_loader
        elif split == "train":  # in case val_loader is None
            data_loader = self.train_loader_x

        print(f"Calculate var and mean on the *{split}* set")        

        # Initialize variables for accumulating mean
        total_samples = 0
        running_mean = None

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)

            visual_feats = torch.stack([res.visual_feat.permute(1, 0, 2) for res in self.model.image_encoder.transformer.resblocks], dim=1)

            # Calculate mean for the current batch
            batch_mean = visual_feats.mean(dim=0)
            
            # Update running mean
            if running_mean is None:
                running_mean = batch_mean
            else:
                running_mean = running_mean * (total_samples / (total_samples + visual_feats.size(0))) + batch_mean * (visual_feats.size(0) / (total_samples + visual_feats.size(0)))

            total_samples += visual_feats.size(0)

        mean_visual_feat = running_mean.cuda()

        # Extract model name from save_path (e.g., 'vitb16' from './stats/vitb16/')
        model_name = save_path.rstrip('/').split('/')[-1]

        print(f"******Saving means embedding to {save_path}*********")
        torch.save(mean_visual_feat, save_path + "V_means_{}_{}_clean.pt".format(model_name, split))

        self.save_and_compute_var(split, mean_visual_feat, data_loader, save_path)

    @torch.no_grad()
    def save_and_compute_var(self, split=None, mean_visual_feat=None, data_loader=None, save_path='./stats/vitb32/'):
        """
        Saving feature maps (i.e. tokens from transformer) and calculating variance in batches
        """
        self.set_model_mode("eval")
        self.evaluator.reset()

        # Initialize variables for accumulating variance
        total_samples = 0
        sum_squared_diff = None

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)

            visual_feats = torch.stack([res.visual_feat.permute(1, 0, 2) for res in self.model.image_encoder.transformer.resblocks], dim=1)

            # Calculate squared difference for the current batch
            batch_squared_diff = (visual_feats - mean_visual_feat).pow(2).sum(dim=0)

            # Update sum of squared differences
            if sum_squared_diff is None:
                sum_squared_diff = batch_squared_diff
            else:
                sum_squared_diff += batch_squared_diff

            total_samples += visual_feats.size(0)

            # Process output and labels (Optional, remove if not needed)
            self.evaluator.process(output, label)

        # Calculate variance
        var_visual_feat = sum_squared_diff / (total_samples - 1)  # Use (total_samples - 1) for sample variance
        var_visual_feat = var_visual_feat.cuda()

        # Extract model name from save_path (e.g., 'vitb16' from './stats/vitb16/')
        model_name = save_path.rstrip('/').split('/')[-1]
        
        print(f"******Saving vars embedding to {save_path}*********")
        torch.save(var_visual_feat, save_path + "V_vars_{}_{}_clean.pt".format(model_name, split))

        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        return list(results.values())[0]