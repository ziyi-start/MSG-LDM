import os
import random

import torch
from torch.utils.data import Dataset

from monai import transforms
from tqdm import tqdm
from multiprocessing.pool import ThreadPool


val_brats_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["t1", "t2", "t1ce", "flair"], image_only=True, allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["t1", "t2", "t1ce", "flair"], allow_missing_keys=True),
        transforms.EnsureTyped(keys=["t1", "t2", "t1ce", "flair"]),
        transforms.Orientationd(keys=["t1", "t2", "t1ce", "flair"], axcodes="RAI", allow_missing_keys=True),
        transforms.CropForegroundd(keys=["t1", "t2", "t1ce", "flair"], source_key="t1", allow_missing_keys=True),
        transforms.SpatialPadd(keys=["t1", "t2", "t1ce", "flair"], spatial_size=(192,192, 155), allow_missing_keys=True),
        transforms.RandSpatialCropd( keys=["t1", "t2", "t1ce", "flair"],
            roi_size=(192, 192, 155),
            random_center=True,
            random_size=False,
        ),
        transforms.ScaleIntensityRangePercentilesd(keys=["t1", "t2", "t1ce", "flair"], lower=0, upper=100, b_min=-1, b_max=1),
    ]
)


def _list_subjects(data_path):
    return [s for s in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, s))]


def _split_subjects(subjects, val_ratio=0.1, seed=42):
    """Deterministic subject-level train/val split.

    The same seed and ratio must be used by CustomTrain and CustomVal so that
    the two subsets are complementary and the split is reproducible.
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


def get_brats_dataset(data_path, phase, subjects=None):
    data = []
    for subject in os.listdir(data_path):
        if subjects is not None and subject not in subjects:
            continue
        sub_path = os.path.join(data_path, subject)
        if not os.path.isdir(sub_path):
            continue
        ext = ".nii.gz" if os.path.exists(os.path.join(sub_path, f"{subject}_t1.nii.gz")) else ".nii"
        t1 = os.path.join(sub_path, f"{subject}_t1{ext}")
        t2 = os.path.join(sub_path, f"{subject}_t2{ext}")
        t1ce = os.path.join(sub_path, f"{subject}_t1ce{ext}")
        flair = os.path.join(sub_path, f"{subject}_flair{ext}")

        data.append({"t1": t1, 't2': t2, 't1ce': t1ce, 'flair': flair})

    return data


class CustomBase(Dataset):

    def __init__(self, data_path, phase, slice_step=1, subjects=None):
        super().__init__()
        self.slice_step = slice_step
        self.slices_per_mod = 120 // slice_step
        self.samples_per_vol = 4 * self.slices_per_mod
        dataset = get_brats_dataset(data_path, phase=phase, subjects=subjects)

        def process_data(data):
            images = val_brats_transforms(data, threading=True)
            return images['t1'], images['t2'], images['t1ce'], images['flair']

        with ThreadPool(10) as pool:
            results = list(tqdm(pool.imap(process_data, dataset), total=len(dataset)))
        self.data = results
        self.phase = phase

    def __len__(self):
        return len(self.data) * self.samples_per_vol

    def __getitem__(self, i):
        vol_idx = i // self.samples_per_vol
        mod_idx = (i % self.samples_per_vol) // self.slices_per_mod  # 0:t1, 1:t2, 2:t1ce, 3:flair
        slice_offset = (i % self.slices_per_mod) * self.slice_step
        slice_idx = 18 + slice_offset
        image = self.data[vol_idx][mod_idx].permute(3, 0, 1, 2)[slice_idx]
        return {'image': image, 'modality': mod_idx}


class CustomTrain(CustomBase):
    def __init__(self, data_path, val_ratio=0.1, seed=42, **kwargs):
        train_subjects, _ = _split_subjects(_list_subjects(data_path), val_ratio, seed)
        super().__init__(data_path=data_path, phase='train', slice_step=1, subjects=train_subjects)


class CustomVal(CustomBase):
    def __init__(self, data_path, val_ratio=0.1, seed=42, **kwargs):
        _, val_subjects = _split_subjects(_list_subjects(data_path), val_ratio, seed)
        super().__init__(data_path=data_path, phase='val', slice_step=4, subjects=val_subjects)


class CustomTest(CustomBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, phase='val', slice_step=4)
