import os
import random

import torch
from torch.utils.data import Dataset
import numpy as np
from monai import transforms
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

image_size = 192
target_slices = 155
SLICE_START = int(os.environ.get("SLICE_START", "70"))
SLICE_END = int(os.environ.get("SLICE_END", "90"))


def _build_transforms():
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=["t1", "t2", "t1ce", "flair", "seg"], image_only=True, allow_missing_keys=True),
            transforms.EnsureChannelFirstd(keys=["t1", "t2", "t1ce", "flair", "seg"], allow_missing_keys=True),
            transforms.EnsureTyped(keys=["t1", "t2", "t1ce", "flair", "seg"]),
            transforms.Orientationd(keys=["t1", "t2", "t1ce", "flair", "seg"], axcodes="RAI", allow_missing_keys=True),
            transforms.CropForegroundd(keys=["t1", "t2", "t1ce", "flair", "seg"], source_key="t1", allow_missing_keys=True),
            transforms.SpatialPadd(keys=["t1", "t2", "t1ce", "flair", "seg"], spatial_size=(image_size, image_size, target_slices), allow_missing_keys=True),
            transforms.CenterSpatialCropd(keys=["t1", "t2", "t1ce", "flair", "seg"], roi_size=(image_size, image_size, target_slices)),
            transforms.ScaleIntensityRangePercentilesd(keys=["t1", "t2", "t1ce", "flair"], lower=0, upper=100, b_min=-1, b_max=1),
        ]
    )


transforms_ = _build_transforms()


def get_brats_dataset(data_path, phase, subjects=None):
    data = []
    for subject in os.listdir(data_path):
        if subjects is not None and subject not in subjects:
            continue
        sub_path = os.path.join(data_path, subject)
        if not os.path.isdir(sub_path):
            continue

        ext = ".nii.gz" if os.path.exists(os.path.join(sub_path, f"{subject}_seg.nii.gz")) else ".nii"
        if not os.path.exists(os.path.join(sub_path, f"{subject}_seg{ext}")):
            continue
        t1 = os.path.join(sub_path, f"{subject}_t1{ext}")
        t2 = os.path.join(sub_path, f"{subject}_t2{ext}")
        t1ce = os.path.join(sub_path, f"{subject}_t1ce{ext}")
        flair = os.path.join(sub_path, f"{subject}_flair{ext}")
        seg = os.path.join(sub_path, f"{subject}_seg{ext}")
        data.append({"t1": t1, "t2": t2, "t1ce": t1ce, "flair": flair, "seg": seg,
                     "name": subject})
    return data


def _list_subjects(data_path):
    return [s for s in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, s))]


def _split_subjects(subjects, val_ratio=0.1, seed=42):
    """Deterministic subject-level train/val split.

    CustomTrain and CustomVal must use the same seed and ratio so that the
    two subsets are complementary and the split is reproducible.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")
    subjects = sorted(subjects)
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_ratio)))
    val = set(subjects[:n_val])
    train = set(subjects[n_val:])
    return train, val


class CustomBase(Dataset):
    def __init__(self, data_path, phase, subjects=None):
        super().__init__()
        dataset = get_brats_dataset(data_path, phase=phase, subjects=subjects)
        self.subjects = [d["name"] for d in dataset]
        xforms = transforms_

        def process_data(data):
            images = xforms(data, threading=True)
            return (
                images["t1"].half(),
                images["t2"].half(),
                images["t1ce"].half(),
                images["flair"].half(),
                images["seg"].to(torch.uint8),
            )

        with ThreadPool(10) as pool:
            results = list(tqdm(pool.imap(process_data, dataset), total=len(dataset)))
        self.data = results
        self.phase = phase

        self.valid_pairs = []
        for vi in range(len(self.data)):
            for si in range(SLICE_START, SLICE_END + 1):
                self.valid_pairs.append((vi, si))
        print(f"[SliceSampler] {len(self.valid_pairs)} slices from {len(self.data)} volumes "
              f"({SLICE_START}~{SLICE_END}, {SLICE_END - SLICE_START + 1} per vol)")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, i):
        vol_idx, slice_idx = self.valid_pairs[i]

        t1, t2, t1ce, flair, seg = self.data[vol_idx]
        t1 = t1.permute(3, 0, 1, 2)[slice_idx].float()
        t2 = t2.permute(3, 0, 1, 2)[slice_idx].float()
        t1ce = t1ce.permute(3, 0, 1, 2)[slice_idx].float()
        flair = flair.permute(3, 0, 1, 2)[slice_idx].float()
        npmask = seg.permute(3, 0, 1, 2)[slice_idx].numpy()

        WT_Label = npmask.copy()
        WT_Label[npmask == 1] = 1.
        WT_Label[npmask == 2] = 1.
        WT_Label[npmask == 4] = 1.

        TC_Label = npmask.copy()
        TC_Label[npmask == 1] = 1.
        TC_Label[npmask == 2] = 0.
        TC_Label[npmask == 4] = 1.

        ET_Label = npmask.copy()
        ET_Label[npmask == 1] = 0.
        ET_Label[npmask == 2] = 0.
        ET_Label[npmask == 4] = 1.

        BG_Label = 1 - (WT_Label > 0).astype(np.float32)

        nplabel = np.empty((image_size, image_size, 4), dtype=np.float32)
        nplabel[:, :, 0] = BG_Label
        nplabel[:, :, 1] = WT_Label
        nplabel[:, :, 2] = TC_Label
        nplabel[:, :, 3] = ET_Label

        data = {
            "t1": t1,
            "t2": t2,
            "t1ce": t1ce,
            "flair": flair,
            "seg": torch.tensor(nplabel.transpose((2, 0, 1)), dtype=torch.float32),
        }
        if self.phase == "train":
            eps = 1e-5
            data["t1"] = data["t1"] + torch.randn_like(data["t1"]) * eps
            data["t2"] = data["t2"] + torch.randn_like(data["t2"]) * eps
            data["t1ce"] = data["t1ce"] + torch.randn_like(data["t1ce"]) * eps
            data["flair"] = data["flair"] + torch.randn_like(data["flair"]) * eps
        return data


class CustomTrain(CustomBase):
    def __init__(self, data_path, val_ratio=0.1, seed=42, **kwargs):
        train_subjects, _ = _split_subjects(_list_subjects(data_path), val_ratio, seed)
        super().__init__(data_path=data_path, phase="train", subjects=train_subjects, **kwargs)


class CustomVal(CustomBase):
    def __init__(self, data_path, val_ratio=0.1, seed=42, **kwargs):
        _, val_subjects = _split_subjects(_list_subjects(data_path), val_ratio, seed)
        super().__init__(data_path=data_path, phase="val", subjects=val_subjects, **kwargs)


class CustomTest(CustomBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, phase="val", **kwargs)
