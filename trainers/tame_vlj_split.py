import math
import os

import torch
import torchvision.transforms as transforms
from dassl.engine import TRAINER_REGISTRY
from torch.utils.data import DataLoader, Subset

from trainers.tame_vlj import AdvAugMixAugmenter, TAMEVLJ, TransformDataset


def _env_int(name, default=None):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


@TRAINER_REGISTRY.register()
class TAMEVLJ_SPLIT(TAMEVLJ):
    """Shard-only TAMEVLJ runner.

    This keeps trainers/tame_vlj.py untouched and selects a contiguous test
    subset using TAPT_SHARD_INDEX and TAPT_SHARD_COUNT.
    """

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

        if args.RUN:
            print("tapt: ", args.RUN)
            preprocess = transforms.Compose([])
            data_transform = AdvAugMixAugmenter(preprocess, n_views=args.BATCH_SIZE - 1, augmix=False)
            batchsize = 1
        else:
            data_transform = transforms.Compose([])
            batchsize = args.BATCH_SIZE

        adv_dataset_pkl_path = os.path.join(
            self.cfg.AT.ADV_DIR, self.cfg.DATASET.NAME + "_adv_dataset.pkl"
        )
        load_kwargs = {"weights_only": False}
        if os.environ.get("TAPT_SHARD_MMAP", "0").lower() in {"1", "true", "yes"}:
            load_kwargs["mmap"] = True
        adv_dataset_pkl = torch.load(adv_dataset_pkl_path, **load_kwargs)
        print("load adv dataset: {}".format(adv_dataset_pkl_path))

        shard_index = _env_int("TAPT_SHARD_INDEX", 0)
        shard_count = _env_int("TAPT_SHARD_COUNT", 1)
        if shard_count < 1:
            raise ValueError("TAPT_SHARD_COUNT must be >= 1")
        if shard_index < 0 or shard_index >= shard_count:
            raise ValueError(
                f"TAPT_SHARD_INDEX={shard_index} is outside [0, {shard_count})"
            )

        total = len(adv_dataset_pkl)
        shard_size = math.ceil(total / shard_count)
        start = shard_index * shard_size
        end = min(start + shard_size, total)
        if start >= end:
            raise ValueError(
                f"empty shard: index={shard_index}, count={shard_count}, total={total}"
            )

        print(
            "[TAPT_SHARD] index={} count={} start={} end={} samples={} total={}".format(
                shard_index, shard_count, start, end, end - start, total
            )
        )
        shard_dataset = Subset(adv_dataset_pkl, range(start, end))
        val_dataset = TransformDataset(shard_dataset, transform=data_transform)

        return DataLoader(
            val_dataset,
            batch_size=batchsize,
            shuffle=True,
            num_workers=_env_int("TAPT_NUM_WORKERS", 8),
            pin_memory=True,
        )
