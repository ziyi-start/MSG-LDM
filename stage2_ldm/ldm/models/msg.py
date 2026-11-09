import os
import sys
import torch
import pytorch_lightning as pl

_HERE = os.path.dirname(os.path.abspath(__file__))          # stage2_ldm/ldm/models
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_MSG_DIR = os.environ.get("MSG_CODE_DIR", os.path.join(_REPO_ROOT, "stage1_msg"))
sys.path.append(_MSG_DIR)
from disentangle import DisentangleModel


class MSGModel(pl.LightningModule):
    def __init__(self, n_modal=4, in_channels=1, ckpt_path=None, ignore_keys=[]):
        super().__init__()
        self.model = DisentangleModel(modal_num=n_modal, in_channels=in_channels)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("MSGModel: deleting key {}".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print("MSGModel: restored from {} (missing={}, unexpected={})".format(
            path, len(missing), len(unexpected)))
