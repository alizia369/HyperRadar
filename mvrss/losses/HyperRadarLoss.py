import torch
from torch import nn
from torch.nn import functional as F

class HyperRadarConsistency(nn.Module):
    def __init__(self, coherence_weight=5., mv_weight=1., huber_delta=1.):
        super().__init__()
        self.coherence_weight=float(coherence_weight)
        self.mv_weight=float(mv_weight)
        self.huber_delta=float(huber_delta)

    def forward(self, rd_logits, ra_logits):
        rd_profile=rd_logits.softmax(dim=1).amax(dim=3,keepdim=True).flip(dims=(2,3))
        ra_profile=ra_logits.softmax(dim=1).amax(dim=3,keepdim=True)
        if rd_profile.shape != ra_profile.shape:
            raise ValueError(f'RD/RA range profile mismatch: {rd_profile.shape} and {ra_profile.shape}')
        return (self.coherence_weight*F.mse_loss(rd_profile,ra_profile)
                + self.mv_weight*F.huber_loss(rd_profile,ra_profile,delta=self.huber_delta))

class ScaleLoss(nn.Module):
    def __init__(self, loss, scale):
        super().__init__()
        self.loss=loss
        self.scale=float(scale)

    def forward(self,pred,target):
        return self.scale*self.loss(pred,target)
