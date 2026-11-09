# Data

Raw data are not included. Put the BraTS folders here:

```
data/
└── BraTS20/
    ├── TrainingData/
    └── TestData/
```

Every split is a directory of case folders. File names must follow:

```
<case>_t1.nii[.gz]
<case>_t2.nii[.gz]
<case>_t1ce.nii[.gz]
<case>_flair.nii[.gz]
<case>_seg.nii[.gz]      # required by stage1_msg and stage2_ldm
```

The VAE stage only needs the four image modalities; the MSG and LDM stages also read the segmentation mask.

`TrainingData` is split at the subject level into train and validation subsets by the dataset classes (`val_ratio: 0.1`, `seed: 42`). `TestData` is held out and is only read by `stage2_ldm/evaluate.py`.

BraTS2021 uses the same layout. Change the `data_path` entries in the three configs accordingly.
