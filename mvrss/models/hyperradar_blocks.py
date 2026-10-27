import math
import torch
from torch import nn
from torch.nn import functional as F

class LearnableHypergraph(nn.Module):

    def __init__(self, channels=64, hyperedges=8, edge_normalize=False, gamma_init=1e-3):
        super().__init__()
        self.assign = nn.Linear(channels, hyperedges)
        self.transform = nn.Sequential(nn.Linear(channels, channels),nn.GELU(),nn.Linear(channels,channels))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.edge_normalize = edge_normalize

    def forward(self, x):
        # feature map B C H W -> B N C
        b,c,h,w = x.shape
        tokens=x.flatten(2).transpose(1,2)
        incidence=torch.softmax(self.assign(tokens),dim=-1) # per token, across hyperedges
        edges=torch.bmm(incidence.transpose(1,2),tokens)
        if self.edge_normalize:
            edges=edges/incidence.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        edges=self.transform(edges)
        updated=tokens+self.gamma*torch.bmm(incidence,edges)
        return updated.transpose(1,2).reshape(b,c,h,w)


class GeneralizedSinkhorn(nn.Module):

    def __init__(self, epsilon=0.05, tau=0.8, iterations=30,
                 normalize_features=True, row_normalize_transport=False):
        super().__init__()
        if epsilon<=0 or tau<=0 or iterations<1:
            raise ValueError('epsilon and tau must be positive; iterations>=1')
        self.epsilon=float(epsilon)
        self.tau=float(tau)
        self.iterations=int(iterations)
        self.normalize_features=normalize_features
        self.row_normalize_transport=row_normalize_transport

    def forward(self, target,source):
        # [B,N,C] and [B,M,C], source-to-target T@source
        if target.shape[0]!=source.shape[0] or target.shape[-1]!=source.shape[-1]:
            raise ValueError(f'Invalid OT features: {target.shape=} {source.shape=}')
        # Sinkhorn should always use fp32 rather than autocast fp16.
        with torch.autocast(device_type=target.device.type,enabled=False):
            x=target.float()
            y=source.float()
            if self.normalize_features:
                x=F.normalize(x,p=2,dim=-1,eps=1e-6)
                y=F.normalize(y,p=2,dim=-1,eps=1e-6)
            cost=torch.cdist(x,y,p=2).square()
            b,n,m=cost.shape
            log_a=cost.new_full((b,n),-math.log(n))
            log_b=cost.new_full((b,m),-math.log(m))
            log_k=-cost/self.epsilon
            u=torch.zeros_like(log_a)
            v=torch.zeros_like(log_b)
            rho=self.tau/(self.tau+self.epsilon)
            for _ in range(self.iterations):
                u=rho*(log_a-torch.logsumexp(log_k+v.unsqueeze(1),dim=2))
                v=rho*(log_b-torch.logsumexp(log_k+u.unsqueeze(2),dim=1))
            plan=torch.exp(log_k+u.unsqueeze(2)+v.unsqueeze(1))
            # Eq. 4 uses raw T @ X, without mass normalization.
            transported=torch.bmm(plan,source.float())
            if self.row_normalize_transport:
                transported=transported/plan.sum(-1,keepdim=True).clamp_min(1e-8)
        return transported.to(target.dtype)


class CrossViewUOT(nn.Module):

    def __init__(self,channels=64,transport_grid=16,epsilon=0.05,tau=0.8,
                 iterations=30,row_normalize_transport=False):
        super().__init__()
        self.ot=GeneralizedSinkhorn(epsilon,tau,iterations,
                                    row_normalize_transport=row_normalize_transport)
        self.transport_grid=transport_grid
        # Eq.5, beta_pq is a learnable scalar. initialize small for stability.
        self.beta=nn.ParameterDict({name:nn.Parameter(torch.tensor(0.1)) for name in
           ['rd_from_ra','rd_from_ad','ra_from_rd','ra_from_ad']})

    def _prepare(self,x):
        if self.transport_grid is None:return x
        hh,ww=x.shape[-2:]
        n=self.transport_grid
        if hh<=n and ww<=n:return x
        return F.adaptive_avg_pool2d(x,(min(n,hh),min(n,ww)))

    def _move(self,target,source):
        tgt=self._prepare(target)
        src=self._prepare(source)
        transported=self.ot(tgt.flatten(2).transpose(1,2),src.flatten(2).transpose(1,2))
        transported=transported.transpose(1,2).reshape_as(tgt)
        if transported.shape[-2:] != target.shape[-2:]:
            transported=F.interpolate(transported,size=target.shape[-2:],mode='bilinear',align_corners=False)
        return transported

    def forward(self,rd,ra,ad):
        # These are distinct cross-view plans, not self-OT of a concatenated map.
        out_rd=rd + self.beta['rd_from_ra']*self._move(rd,ra) + self.beta['rd_from_ad']*self._move(rd,ad)
        out_ra=ra + self.beta['ra_from_rd']*self._move(ra,rd) + self.beta['ra_from_ad']*self._move(ra,ad)
        return out_rd,out_ra


class ChannelGate(nn.Module):

    def __init__(self,channels=64,reduction=4):
        super().__init__()
        hidden=max(channels//reduction,4)
        self.psi=nn.Sequential(nn.Linear(channels,hidden),nn.ReLU(inplace=True),
                               nn.Linear(hidden,channels))
        nn.init.zeros_(self.psi[-1].weight)
        nn.init.zeros_(self.psi[-1].bias)

    def forward(self,x):
        descriptor=x.mean(dim=(2,3))
        gates=2*torch.sigmoid(self.psi(descriptor)).unsqueeze(-1).unsqueeze(-1)
        return x*gates


