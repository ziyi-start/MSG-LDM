import os

import torch
from torch.utils.data import Dataset

from monai import transforms


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

def get_brats_dataset(data_path, phase):
    data = []
    for subject in os.listdir(data_path):
        sub_path = os.path.join(data_path, subject)
        if os.path.exists(sub_path) == False: continue
        t1 = os.path.join(sub_path, f"{subject}_t1.nii")
        t2 = os.path.join(sub_path, f"{subject}_t2.nii")
        t1ce = os.path.join(sub_path, f"{subject}_t1ce.nii")
        flair = os.path.join(sub_path, f"{subject}_flair.nii")

        data.append({"t1": t1, 't2': t2, 't1ce': t1ce, 'flair': flair})
        # break

    return data

from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import random
class CustomBase(Dataset):
    def __init__(self, data_path, phase):
        super().__init__()
        dataset = get_brats_dataset(data_path,phase=phase)
        # self.t1_list = []
        # self.t2_list = []

        # 定义一个函数，用于多线程执行的任务
        def process_data(data):
            images = val_brats_transforms(data, threading=True)
            return images['t1'], images['t2'], images['t1ce'], images['flair']

        # 创建 ThreadPool 对象，并指定线程数量
        with ThreadPool(10) as pool:
            # 提交任务给线程池
            results = list(tqdm(pool.imap(process_data, dataset), total=len(dataset)))
        self.data = results
        self.phase = phase

    def __len__(self):
        return len(self.data) * 120 * 4

    def __getitem__(self, i):

        data = {'image': self.data[i // 500][(i % (4 * 120)) // 120].permute(3, 0, 1, 2)[18 + i % 120]}

        return data

class CustomTrain(CustomBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, phase='train')



class CustomTest(CustomBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, phase='val')
