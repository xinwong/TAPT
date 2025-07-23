import argparse
import torch
import os

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r
import datasets.pug

import trainers.adv_vli
import trainers.adv_vlj


from pdb import set_trace as stx

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.advdata_dir:
        cfg.AT.ADV_DIR = args.advdata_dir


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN
    
    # TAPT args
    cfg.TAPT = CN()
    cfg.TAPT.LOADER = True   # Use TAPT Dataloader. (Just for sanity check)
    cfg.TAPT.RUN = True  # Run TAPT using TAPT dataloader
    cfg.TAPT.LR = 4e-2   # Learning rate for TAPT
    cfg.TAPT.COCOOP = False
    cfg.TAPT.ALIGN_LAYER_FROM = 0
    cfg.TAPT.ALIGN_LAYER_TO = 9
    cfg.TAPT.TTA_STEPS = 1
    cfg.TAPT.DISTR_ALIGN = False
    cfg.TAPT.TAPT_THRESHOLD = 0.1
    cfg.TAPT.ALIGN_THRESHOLD = 0.1
    cfg.TAPT.TAPT_LOSS = True
    cfg.TAPT.DISTR_LOSS_W = 100.0
    cfg.TAPT.BATCH_SIZE = 64
    cfg.TAPT.VIS_MEANS = './statistics/imagenet_VLI_means_adv.pt'
    cfg.TAPT.VIS_VARS = './statistics/imagenet_VLI_means_adv.pt'
    cfg.TAPT.VIS_MEANS_CLEAN = './statistics/imagenet_VLI_means_clean.pt'
    cfg.TAPT.VIS_VARS_CLEAN = './statistics/imagenet_VLI_vars_clean.pt'
    cfg.TAPT.GAMMA= 0.5
    cfg.TAPT.RESET= 1
    cfg.AT = CN()
    cfg.AT.TEST = CN()
    cfg.AT.TEST.EPS = 4
    cfg.AT.TEST.ALPHA = 1
    cfg.AT.TEST.STEPS = 100
    cfg.AT.ADV_DIR = ''
    cfg.AT.ALIGN_TYPE = ''
    cfg.AT.ADVALIGN = True
    
    # Config for VLI
    cfg.TRAINER.VLPROMPTALIGN = CN()
    cfg.TRAINER.VLPROMPTALIGN.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.VLPROMPTALIGN.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.VLPROMPTALIGN.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.VLPROMPTALIGN.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.VLPROMPTALIGN.PROMPT_DEPTH_VISION = 9 # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.TRAINER.VLPROMPTALIGN.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.TAPT = 'I'
    cfg.DATASET.VARIANT = 'Worlds'  # Added for PUG dataset variants

    # Config for VLJ
    cfg.TRAINER.PROMPTALIGN = CN()
    cfg.TRAINER.PROMPTALIGN.N_CTX = 2  # number of context vectors
    cfg.TRAINER.PROMPTALIGN.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.PROMPTALIGN.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTALIGN.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.TAPT = 'I'


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg

def configure_attack(cfg):
    test_eps    = cfg.AT.TEST.EPS / 255.0
    test_alpha  = cfg.AT.TEST.ALPHA  / 255.0
    test_step   = cfg.AT.TEST.STEPS

    print("Test_eps:{}, Test_alpha:{}, Test_step: {}".format(test_eps, test_alpha, test_step))

    return test_eps, test_alpha, test_step

def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    if args.tapt:
        assert args.tapt == cfg.TAPT.RUN, "TAPT flag in args and config mismatch"
    trainer = build_trainer(cfg)
    configure_attack(cfg)

    if args.tapt:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        results = trainer.tapt()
        print()
        print("\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
        for id in results.keys():
            print("{}".format(id), end="	")
        print("\n")
        for id in results.keys():
            print("{:.2f}".format(results[id][0]), end="	")
        print("\n")
        return
    else:
        raise ValueError("Invalid mode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--advdata-dir",
        type=str,
        default=None,
        help="load adv data from this directory for align prompt mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument('--tapt', action='store_true', default=True, help='run test-time adversarial prompt tuning')
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
