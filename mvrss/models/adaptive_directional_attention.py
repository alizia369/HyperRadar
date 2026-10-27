import torch
from torch import nn
from torch.nn import functional as F
from operator import itemgetter
import torchvision.ops
import cripser
import ot


# =========================================================
# Base modules
# =========================================================

class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    ):
        super().__init__()

        assert isinstance(kernel_size, (tuple, int))

        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=True
        )
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(
            in_channels,
            1 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=True
        )
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=bias
        )

    def forward(self, x):
        offset = self.offset_conv(x)
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=modulator,
            stride=self.stride,
        )
        return x


def exists(val):
    return val is not None


def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))


def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)


def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []
    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)

    return permutations


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Sequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g in self.blocks:
            x = x + f(x)
            x = x + g(x)
        return x


class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn, dim=64, kernel_size=3):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation
        self.deform = DeformableConv2d(
            dim, dim, (kernel_size, kernel_size),
            padding=(kernel_size // 2, kernel_size // 2)
        )

    def forward(self, x, **kwargs):
        x = self.deform(x)
        axial = x.permute(*self.permutation).contiguous()
        shape = axial.shape
        *_, t, d = shape

        axial = axial.reshape(-1, t, d)
        axial = self.fn(axial, **kwargs)

        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads=None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))
        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda z: z.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)
        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out


# =========================================================
# Cripser + OT topology gate
# =========================================================

class CripserOTTopoGate(nn.Module):
    """
    Topology gate before ADA.

    New persistence strategy:
    - no hard top-k selection
    - importance is based on distance from the diagonal
    - points near diagonal = low persistence = low importance
    - points far from diagonal = high persistence = high importance
    """

    def __init__(
        self,
        channels,
        homology_dim=1,
        maxdim=1,
        topk_pairs=8,              # kept only for compatibility, not used now
        downsample=4,
        alpha=8.0,
        tau=0.0,
        ot_reg=0.1,
        smooth_train=True,
        temperature=0.15,
        hard_prune_eval=False,
        keep_ratio=0.5,
        use_post=True,
        min_diag_distance=1e-6,
        distance_power=2.0,
        eps=1e-8
    ):
        super().__init__()
        self.channels = channels
        self.homology_dim = homology_dim
        self.maxdim = maxdim
        self.topk_pairs = topk_pairs
        self.downsample = downsample
        self.ot_reg = ot_reg
        self.smooth_train = smooth_train
        self.temperature = temperature
        self.hard_prune_eval = hard_prune_eval
        self.keep_ratio = keep_ratio
        self.min_diag_distance = min_diag_distance
        self.distance_power = distance_power
        self.eps = eps

        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))

        self.use_post = use_post
        if self.use_post:
            self.gamma = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.post = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU()
            )

        self.last_scores = None
        self.last_gates = None

    def _normalize_map(self, fm):
        mn = fm.min()
        mx = fm.max()
        return (fm - mn) / (mx - mn + self.eps)

    def _extract_birth_death_points(self, ph, target_device):
        """
        ph rows are expected to start as:
        [dim, birth, death, ...]

        Returns:
            points:    [N, 2] = [birth, death]
            distances: [N]    = distance from diagonal
        """
        if ph is None or ph.numel() == 0:
            return None, None

        ph = ph.to(device=target_device)

        dims = ph[:, 0].long()
        diag = ph[dims == self.homology_dim][:, 1:3]   # [birth, death]

        if diag.numel() == 0:
            return None, None

        finite = torch.isfinite(diag).all(dim=1)
        diag = diag[finite]
        if diag.numel() == 0:
            return None, None

        births = diag[:, 0]
        deaths = diag[:, 1]

        valid = torch.isfinite(births) & torch.isfinite(deaths) & (deaths > births)
        births = births[valid]
        deaths = deaths[valid]

        if births.numel() == 0:
            return None, None

        points = torch.stack([births, deaths], dim=1)

        # distance from diagonal death=birth
        distances = (deaths - births).abs() / (2.0 ** 0.5)

        valid_dist = torch.isfinite(distances) & (distances > self.min_diag_distance)
        points = points[valid_dist]
        distances = distances[valid_dist]

        if distances.numel() == 0:
            return None, None

        return points, distances

    def _make_masses(self, distances):
        """
        More persistent points (farther from diagonal) get larger OT mass.
        """
        if distances is None or distances.numel() == 0:
            return None

        weights = distances ** self.distance_power
        masses = weights / (weights.sum() + self.eps)
        return masses

    def _sinkhorn_transport_and_fuse(self, pts_v, mass_v, pts_t, mass_t):
        if pts_v is None or pts_t is None:
            return None, None, None
        if pts_v.numel() == 0 or pts_t.numel() == 0:
            return None, None, None

        device = pts_v.device
        dtype = pts_v.dtype

        pts_v = pts_v.to(device=device, dtype=dtype)
        pts_t = pts_t.to(device=device, dtype=dtype)
        mass_v = mass_v.to(device=device, dtype=dtype)
        mass_t = mass_t.to(device=device, dtype=dtype)

        M = torch.cdist(pts_v, pts_t, p=2).pow(2)
        M = M / (M.max().detach() + self.eps)

        P = ot.bregman.sinkhorn_log(
            mass_v,
            mass_t,
            M,
            reg=self.ot_reg,
            numItermax=2000,
            stopThr=1e-7,
            warn=False,
            log=False
        )

        if not torch.is_tensor(P):
            P = torch.tensor(P, device=device, dtype=dtype)
        else:
            P = P.to(device=device, dtype=dtype)

        row_mass = P.sum(dim=1)
        aligned_t = P @ pts_t
        aligned_t = aligned_t / (row_mass.unsqueeze(1) + self.eps)

        fused_points = 0.5 * (pts_v + aligned_t)
        fused_mass = row_mass / (row_mass.sum() + self.eps)
        ot_cost = (P * M).sum()

        return fused_points, fused_mass, ot_cost

    def _score_from_fused_points(self, fused_points, fused_mass, ot_cost, ref_tensor):
        """
        Channel importance is based on fused point distance from diagonal.
        """
        if fused_points is None or fused_mass is None:
            return ref_tensor.new_tensor(0.0)

        births = fused_points[:, 0]
        deaths = fused_points[:, 1]

        fused_distances = (deaths - births).abs() / (2.0 ** 0.5)
        fused_distances = fused_distances.clamp(min=0)

        if fused_distances.numel() == 0:
            return ref_tensor.new_tensor(0.0)

        if self.training and self.smooth_train and fused_distances.numel() > 1:
            weights = torch.softmax(fused_distances / self.temperature, dim=0)
            topo_strength = (weights * fused_distances).sum()
        else:
            topo_strength = (fused_mass * fused_distances).sum()

        score = topo_strength / (1.0 + ot_cost)
        return score.to(device=ref_tensor.device, dtype=ref_tensor.dtype)

    def _compute_channel_score(self, fm, out_dtype, out_device):
        """
        fm: [H, W] one feature map
        returns scalar on original model dtype/device
        """
        fm = self._normalize_map(fm)

        ph_v = cripser.compute_ph_torch(fm, maxdim=self.maxdim, filtration="V")
        ph_t = cripser.compute_ph_torch(fm, maxdim=self.maxdim, filtration="T")

        pts_v, dist_v = self._extract_birth_death_points(ph_v, target_device=fm.device)
        pts_t, dist_t = self._extract_birth_death_points(ph_t, target_device=fm.device)

        if pts_v is None or pts_t is None:
            return torch.zeros((), device=out_device, dtype=out_dtype)

        mass_v = self._make_masses(dist_v)
        mass_t = self._make_masses(dist_t)

        fused_points, fused_mass, ot_cost = self._sinkhorn_transport_and_fuse(
            pts_v, mass_v, pts_t, mass_t
        )

        score = self._score_from_fused_points(
            fused_points, fused_mass, ot_cost, ref_tensor=fm
        )

        return score.to(device=out_device, dtype=out_dtype)

    def _apply_eval_hard_prune(self, gates):
        if self.training or not self.hard_prune_eval:
            return gates

        B, C, _, _ = gates.shape
        k = max(1, int(round(C * self.keep_ratio)))

        flat = gates.view(B, C)
        pruned = torch.zeros_like(flat)

        for b in range(B):
            idx = flat[b].topk(k=k, largest=True).indices
            pruned[b, idx] = flat[b, idx]

        return pruned.view(B, C, 1, 1)

    def forward(self, x):
        orig_dtype = x.dtype
        orig_device = x.device

        B, C, H, W = x.shape
        assert C == self.channels, f"Expected {self.channels} channels, got {C}"

        topo_x = x
        if self.downsample > 1:
            topo_x = F.avg_pool2d(topo_x, kernel_size=self.downsample, stride=self.downsample)

        all_scores = []
        for b in range(B):
            sample_scores = []
            for c in range(C):
                fm = topo_x[b, c]
                score = self._compute_channel_score(
                    fm,
                    out_dtype=orig_dtype,
                    out_device=orig_device
                )
                sample_scores.append(score)

            sample_scores = torch.stack(sample_scores, dim=0)
            all_scores.append(sample_scores)

        scores = torch.stack(all_scores, dim=0).to(device=orig_device, dtype=orig_dtype)

        mean = scores.mean(dim=1, keepdim=True)
        std = scores.std(dim=1, unbiased=False, keepdim=True) + self.eps
        scores_norm = (scores - mean) / std

        alpha = self.alpha.to(device=orig_device, dtype=orig_dtype)
        tau = self.tau.to(device=orig_device, dtype=orig_dtype)

        gates = torch.sigmoid(alpha * (scores_norm - tau))
        gates = gates.view(B, C, 1, 1).to(device=orig_device, dtype=orig_dtype)

        gates = self._apply_eval_hard_prune(gates)
        gates = gates.to(device=orig_device, dtype=orig_dtype)

        x = (x * gates).to(device=orig_device, dtype=orig_dtype)

        if self.use_post:
            gamma = self.gamma.to(device=orig_device, dtype=orig_dtype)
            x = x + gamma * self.post(x)

        self.last_scores = scores.detach()
        self.last_gates = gates.detach()

        return x


# =========================================================
# ADA with topology gate
# =========================================================

class ADA(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads=8,
        dim_heads=None,
        dim_index=1,
        deform_k=[3, 3, 3, 3, 3, 3, 3, 3],

        use_tda=True,
        topo_homology_dim=1,
        topo_maxdim=1,
        topo_topk_pairs=8,          # compatibility only, ignored
        topo_downsample=4,
        topo_alpha=8.0,
        topo_tau=0.0,
        topo_ot_reg=0.1,
        topo_smooth_train=True,
        topo_temperature=0.15,
        topo_hard_prune_eval=False,
        topo_keep_ratio=0.5,
        topo_use_post=True,
        topo_min_diag_distance=1e-6,
        topo_distance_power=2.0
    ):
        super().__init__()

        permutations = calculate_permutations(2, dim_index)
        self.pos_emb = nn.Identity()

        self.use_tda = use_tda
        if self.use_tda:
            self.tda = CripserOTTopoGate(
                channels=dim,
                homology_dim=topo_homology_dim,
                maxdim=topo_maxdim,
                topk_pairs=topo_topk_pairs,
                downsample=topo_downsample,
                alpha=topo_alpha,
                tau=topo_tau,
                ot_reg=topo_ot_reg,
                smooth_train=topo_smooth_train,
                temperature=topo_temperature,
                hard_prune_eval=topo_hard_prune_eval,
                keep_ratio=topo_keep_ratio,
                use_post=topo_use_post,
                min_diag_distance=topo_min_diag_distance,
                distance_power=topo_distance_power
            )

        layers = nn.ModuleList([])
        for mb in range(depth):
            attn_functions = nn.ModuleList([
                PermuteToFrom(
                    permutation,
                    PreNorm(dim, SelfAttention(dim, heads, dim_heads)),
                    dim=dim,
                    kernel_size=deform_k[mb]
                )
                for permutation in permutations
            ])
            layers.append(attn_functions)

        self.layers = Sequential(layers)

    def forward(self, x):
        if self.use_tda:
            x = self.tda(x)

        x = self.pos_emb(x)
        return self.layers(x)