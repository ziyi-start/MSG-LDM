import os

import torch
from torch.utils.data import Dataset
import numpy as np
from monai import transforms

image_size = 192
val_brats_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["t1", "t2", "t1ce", "flair", "seg"], image_only=True, allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["t1", "t2", "t1ce", "flair", "seg"], allow_missing_keys=True),
        transforms.EnsureTyped(keys=["t1", "t2", "t1ce", "flair", "seg"]),
        transforms.Orientationd(keys=["t1", "t2", "t1ce", "flair", "seg"], axcodes="RAI", allow_missing_keys=True),
        transforms.CropForegroundd(keys=["t1", "t2", "t1ce", "flair", "seg"], source_key="t1", allow_missing_keys=True),
        transforms.SpatialPadd(keys=["t1", "t2", "t1ce", "flair", "seg"], spatial_size=(image_size,image_size, 155), allow_missing_keys=True),

        transforms.CenterSpatialCropd(
         keys=["t1", "t2", "t1ce", "flair", "seg"],
         roi_size=(192,192,155),
        ),

        transforms.ScaleIntensityRangePercentilesd(keys=["t1", "t2", "t1ce", "flair"], lower=0, upper=100, b_min=-1, b_max=1),
    ]
)

def get_brats_dataset(data_path, phase):
    data = []
    for i, subject in enumerate(os.listdir(data_path)):
        sub_path = os.path.join(data_path, subject)
        if os.path.exists(sub_path) == False: continue
        if os.path.exists(os.path.join(sub_path, f"{subject}_seg.nii")) == False: continue
        t1 = os.path.join(sub_path, f"{subject}_t1.nii")
        t2 = os.path.join(sub_path, f"{subject}_t2.nii")
        t1ce = os.path.join(sub_path, f"{subject}_t1ce.nii")
        flair = os.path.join(sub_path, f"{subject}_flair.nii")
        seg = os.path.join(sub_path, f"{subject}_seg.nii")

        data.append({"t1": t1, 't2': t2, 't1ce': t1ce, 'flair': flair, 'seg': seg})
        # break
        # if i == 2:
        #     break

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
            return images['t1'], images['t2'], images['t1ce'], images['flair'], images['seg']

        # 创建 ThreadPool 对象，并指定线程数量
        with ThreadPool(10) as pool:
            # 提交任务给线程池
            results = list(tqdm(pool.imap(process_data, dataset), total=len(dataset)))
        self.data = results

        self.phase = phase

    def __len__(self):
        # return 125
        return len(self.data) * 120

    def __getitem__(self, i):

        data = {'t1': self.data[i // 120][0].permute(3,0,1,2)[18+i % 120],
                't2': self.data[i // 120][1].permute(3,0,1,2)[18+i % 120],
                't1ce': self.data[i // 120][2].permute(3,0,1,2)[18+i % 120],
                'flair': self.data[i // 120][3].permute(3,0,1,2)[18+i % 120],
                'seg': self.data[i // 120][4].permute(3,0,1,2)[18+i % 120],}
        npmask = data['seg'].numpy()

        # 原始标签映射
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

        # 新增背景通道 = 1 - (WT_Label>0)
        BG_Label = 1 - (WT_Label > 0).astype(np.float32)

        # 拼成4通道 (背景, WT, TC, ET)
        nplabel = np.empty((image_size, image_size, 4), dtype=np.float32)
        nplabel[:, :, 0] = BG_Label
        nplabel[:, :, 1] = WT_Label
        nplabel[:, :, 2] = TC_Label
        nplabel[:, :, 3] = ET_Label

        # 转换为 (C,H,W)
        nplabel = nplabel.transpose((2, 0, 1))
        data['seg'] = torch.tensor(nplabel, dtype=torch.float32)

        # del data['seg']
        return data

class CustomTrain(CustomBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, phase='train')



class CustomTest(CustomBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, phase='val')




