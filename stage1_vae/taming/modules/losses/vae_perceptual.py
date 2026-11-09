import torch
import torch.nn as nn

from taming.modules.losses.lpips import LPIPS


class L1PerceptualKLLoss(nn.Module):
    def __init__(self, kl_weight=1.0e-4, perceptual_weight=1.0, logvar_init=0.0):
        super().__init__()
        self.kl_weight = kl_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def forward(self, inputs, reconstructions, posteriors, global_step, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0.:
            if inputs.shape[1] >= 3:
                p_loss = self.perceptual_loss(inputs[:, :3].contiguous(), reconstructions[:, :3].contiguous())
            else:
                p_loss = self.perceptual_loss(
                    inputs[:, :1].repeat(1, 3, 1, 1).contiguous(),
                    reconstructions[:, :1].repeat(1, 3, 1, 1).contiguous()
                )
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        loss = nll_loss + self.kl_weight * kl_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/rec_loss".format(split): rec_loss.detach().mean(),
               "{}/p_loss".format(split): p_loss.detach().mean(),
               "{}/nll_loss".format(split): nll_loss.detach().mean(),
               "{}/kl_loss".format(split): kl_loss.detach().mean(),
               "{}/logvar".format(split): self.logvar.detach(),
               }
        return loss, log
