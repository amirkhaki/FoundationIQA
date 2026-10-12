"""DINO-only full-reference IQA architecture.

This is the spatial (DINOv2) expert of ``FoundationHybrid`` factored out into a
standalone model: no CNN/texture expert and no gate. The per-view score is the
DINO score itself, and the views (global / center / texture) are combined with
``view_weights`` exactly like in the hybrid model, so with default arguments it
reproduces the hybrid's "DINO only" ablation (``A3_dino_only``).

Scoring pipeline (per view)::

    for every selected DINOv2 block l:
        dists_l     : DISTS-style SSIM of the patch-token feature maps
        patch_cos_l : mean / worst-k / mix of the per-patch cosine similarity
        cls_l       : cosine similarity of the CLS tokens
        layer_l     = w_dists * dists_l + w_patch * patch_cos_l + w_cls * cls_l
    s_view = sum_l(layer_weight_l * layer_l) / sum_l(layer_weight_l)

    score = sum_view(view_weight * s_view)
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.default_model_configs import DEFAULT_CONFIGS

from .arch import DINOv2SpatialExtractor

_DEFAULT_LAYER_INDICES = {
    'vits': [0, 3, 6, 9, 11],
    'vitb': [0, 3, 6, 9, 11],
    'vitl': [0, 5, 11, 17, 23],
    'vitg': [0, 9, 19, 29, 39],
}


def default_dino_layer_indices(model_name):
    for key, idx in _DEFAULT_LAYER_INDICES.items():
        if key in model_name:
            return list(idx)
    raise ValueError(f"No default layer indices for '{model_name}'; pass dino_layer_indices explicitly.")


@ARCH_REGISTRY.register()
class FoundationDino(nn.Module):
    """DINOv2-only FR-IQA model. ``forward(ref, dist)`` returns a (B,) score (higher = better)."""

    def __init__(self,
                 dino_model_name="dinov2_vitb14",
                 device=None,
                 xi=1e-6,
                 # DINO score
                 dino_score_terms=('dists', 'patch_cos', 'cls'),
                 dino_term_weights=(0.45, 0.45, 0.10),     # (dists, patch_cos, cls)
                 dino_patch_cos_mode='both',               # 'mean' | 'worst' | 'both'
                 worst_k_ratio=0.10,
                 mean_worst_mix_alpha=0.5,
                 dino_layer_weights=(0.25, 0.30, 0.25, 0.12, 0.08),
                 dino_layer_indices=None,
                 dino_dists_local=False,
                 dino_dists_ws=3,
                 dino_norm_imagenet=True,
                 dino_padding_mode='reflect',
                 dino_resize=None,
                 # Pre-processing (same as the hybrid: Resize(256) + CenterCrop(224))
                 spatial_preprocess=True,
                 spatial_early=True,
                 # Multi-view
                 multiscale=True,
                 views=('global', 'center', 'texture'),
                 view_weights=None,
                 crop_ratio=0.70,
                 crop_selection='max_var_ref',             # 'max_var_ref' | 'max_var_dist' | 'max_diff' | 'min_var'
                 texture_crops_k=1,
                 texture_agg='mean',                       # 'mean' | 'min'
                 texture_var_ws=7,
                 pyramid_mode=False,
                 return_cache=False,
                 **kwargs):
        super().__init__()
        if kwargs:
            warnings.warn(f"FoundationDino ignoring unknown arguments: {sorted(kwargs)}")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.xi = xi

        self.dino_score_terms = list(dino_score_terms)
        self.dino_term_weights = tuple(dino_term_weights)
        self.dino_patch_cos_mode = dino_patch_cos_mode
        self.worst_k_ratio = worst_k_ratio
        self.mean_worst_mix_alpha = mean_worst_mix_alpha
        self.dino_layer_weights = list(dino_layer_weights)
        self.dino_layer_indices = (list(dino_layer_indices) if dino_layer_indices is not None
                                   else default_dino_layer_indices(dino_model_name))
        if len(self.dino_layer_weights) != len(self.dino_layer_indices):
            raise ValueError(f"dino_layer_weights ({len(self.dino_layer_weights)}) and dino_layer_indices "
                             f"({len(self.dino_layer_indices)}) must have the same length")
        self.dino_dists_local = dino_dists_local
        self.dino_dists_ws = dino_dists_ws
        self.dino_norm_imagenet = dino_norm_imagenet
        self.dino_padding_mode = dino_padding_mode
        self.dino_resize = dino_resize

        self.spatial_early = spatial_early
        self.spatial_transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.CenterCrop(224),
        ]) if spatial_preprocess else None

        self.multiscale = multiscale
        self.views = list(views)
        self.view_weights = dict(view_weights) if view_weights is not None \
            else {'global': 0.60, 'center': 0.25, 'texture': 0.15}
        self.crop_ratio = crop_ratio
        self.crop_selection = crop_selection
        self.texture_crops_k = texture_crops_k
        self.texture_agg = texture_agg
        self.texture_var_ws = texture_var_ws
        self.pyramid_mode = pyramid_mode
        self.return_cache = return_cache

        self.dino = DINOv2SpatialExtractor(dino_model_name, self.dino_layer_indices, device=self.device)

    # ------------------------------------------------------------------ DINO score
    def _patch_cos_score(self, mean_cos, cos_flat):
        k_val = max(1, int(cos_flat.shape[1] * self.worst_k_ratio))
        worst_cos = torch.topk(cos_flat, k_val, dim=1, largest=False)[0].mean(dim=1)
        if self.dino_patch_cos_mode == 'mean':
            return mean_cos
        if self.dino_patch_cos_mode == 'worst':
            return worst_cos
        return self.mean_worst_mix_alpha * mean_cos + (1 - self.mean_worst_mix_alpha) * worst_cos

    def _dists_score(self, fr, fd):
        if self.dino_dists_local:
            pool = nn.AvgPool2d(kernel_size=self.dino_dists_ws, stride=1, padding=self.dino_dists_ws // 2)
            mr, md = pool(fr), pool(fd)
            vr = torch.clamp(pool(fr ** 2) - mr ** 2, min=0.0)
            vd = torch.clamp(pool(fd ** 2) - md ** 2, min=0.0)
            cov = pool(fr * fd) - mr * md
        else:
            mr = torch.mean(fr, dim=(2, 3), keepdim=True)
            md = torch.mean(fd, dim=(2, 3), keepdim=True)
            vr = torch.var(fr, dim=(2, 3), unbiased=False, keepdim=True)
            vd = torch.var(fd, dim=(2, 3), unbiased=False, keepdim=True)
            cov = torch.mean((fr - mr) * (fd - md), dim=(2, 3), keepdim=True)
        s_mean = (2 * mr * md + self.xi) / (mr ** 2 + md ** 2 + self.xi)
        s_var = (2 * cov + self.xi) / (vr + vd + self.xi)
        return (s_mean * s_var).mean(dim=(1, 2, 3))

    @torch.no_grad()
    def _compute_dino_score(self, ref, dist):
        """Per-view DINO score (B,), or, with ``return_cache``, the raw per-layer terms."""
        if ref.max() <= 3.0 and ref.min() < -0.1:
            warnings.warn("Input to DINOv2 appears to be normalized (has negative values). Check if this is intended.")

        if self.dino_resize is not None:
            H, W = ref.shape[2:]
            if H < W:
                new_h, new_w = self.dino_resize, int(self.dino_resize * W / H)
            else:
                new_h, new_w = int(self.dino_resize * H / W), self.dino_resize
            ref = F.interpolate(ref, size=(new_h, new_w), mode='bicubic', align_corners=False)
            dist = F.interpolate(dist, size=(new_h, new_w), mode='bicubic', align_corners=False)

        maps_r, cls_r = self.dino(ref, padding_mode=self.dino_padding_mode, dino_norm_imagenet=self.dino_norm_imagenet)
        maps_d, cls_d = self.dino(dist, padding_mode=self.dino_padding_mode, dino_norm_imagenet=self.dino_norm_imagenet)

        B = ref.shape[0]
        zeros = torch.zeros(B, device=ref.device)
        lws = self.dino_layer_weights
        if sum(lws) == 0:
            lws = [1.0 / len(lws)] * len(lws)
        w_dists, w_patch, w_cls = self.dino_term_weights

        weighted, cache = [], []
        for fr, fd, cr, cd, lw in zip(maps_r, maps_d, cls_r, cls_d, lws):
            dists_score = self._dists_score(fr, fd) if 'dists' in self.dino_score_terms else zeros

            mean_cos = cos_flat = None
            patch_cos_score = zeros
            if 'patch_cos' in self.dino_score_terms:
                cos_map = (F.normalize(fr, p=2, dim=1) * F.normalize(fd, p=2, dim=1)).sum(dim=1)
                cos_flat = cos_map.view(B, -1)
                mean_cos = cos_flat.mean(dim=1)
                patch_cos_score = self._patch_cos_score(mean_cos, cos_flat)

            cls_score = zeros
            if 'cls' in self.dino_score_terms:
                cls_score = (F.normalize(cr, p=2, dim=1) * F.normalize(cd, p=2, dim=1)).sum(dim=1)

            layer_score = w_dists * dists_score + w_patch * patch_cos_score + w_cls * cls_score
            weighted.append(lw * layer_score)

            if self.return_cache:
                cache.append({
                    'dists_score': dists_score,
                    'mean_cos': mean_cos if mean_cos is not None else zeros,
                    'cos_flat': cos_flat if cos_flat is not None else torch.zeros((B, 1), device=ref.device),
                    'cls_score': cls_score,
                })

        if self.return_cache:
            return cache
        return torch.stack(weighted, dim=0).sum(dim=0) / sum(lws)

    @torch.no_grad()
    def _view_forward(self, ref, dist):
        if not self.spatial_early and self.spatial_transform is not None:
            ref, dist = self.spatial_transform(ref), self.spatial_transform(dist)
        out = self._compute_dino_score(ref, dist)
        return {'dino': out} if self.return_cache else out

    # ------------------------------------------------------------------ views
    def _texture_var_map(self, ref, dist):
        ws = self.texture_var_ws

        def local_var(x):
            g = x.mean(dim=1, keepdim=True)
            return F.avg_pool2d(g ** 2, ws, 1, ws // 2) - F.avg_pool2d(g, ws, 1, ws // 2) ** 2

        sel = self.crop_selection
        if sel == 'max_var_ref':
            return local_var(ref)
        if sel == 'max_var_dist':
            return local_var(dist)
        if sel == 'max_diff':
            return (ref - dist).abs().mean(dim=1, keepdim=True)
        if sel == 'min_var':
            return -local_var(ref)
        raise ValueError(f"Unknown crop_selection: {sel}")

    def _texture_crops(self, ref, dist, crop_size, k):
        """k batched (ref_crop, dist_crop) pairs; crop j is centred on the j-th strongest peak of each sample."""
        B, _, H, W = ref.shape
        var_map = self._texture_var_map(ref, dist)
        w_map = var_map.shape[-1]
        flat = var_map.reshape(B, -1)
        if k == 1:
            ids = torch.argmax(flat, dim=1, keepdim=True)  # same tie-breaking as the hybrid model
        else:
            ids = torch.topk(flat, k, dim=1)[1]  # (B, k), descending

        crops = []
        for j in range(k):
            refs, dists = [], []
            for i in range(B):
                idx = int(ids[i, j])
                h_idx, w_idx = idx // w_map, idx % w_map
                top = max(0, min(H - crop_size, h_idx - crop_size // 2))
                left = max(0, min(W - crop_size, w_idx - crop_size // 2))
                refs.append(ref[i:i + 1, :, top:top + crop_size, left:left + crop_size])
                dists.append(dist[i:i + 1, :, top:top + crop_size, left:left + crop_size])
            crops.append((torch.cat(refs, dim=0), torch.cat(dists, dim=0)))
        return crops

    def _active_views(self):
        return list(self.views) if self.multiscale else ['global']

    @torch.no_grad()
    def forward(self, ref, dist, **kwargs):
        if self.spatial_early and self.spatial_transform is not None:
            ref, dist = self.spatial_transform(ref), self.spatial_transform(dist)

        views = self._active_views()
        results = {}

        # Global view (optionally at half resolution)
        if self.pyramid_mode:
            ref_g = F.interpolate(ref, scale_factor=0.5, mode='bicubic')
            dist_g = F.interpolate(dist, scale_factor=0.5, mode='bicubic')
        else:
            ref_g, dist_g = ref, dist
        # The global view is always computed (it is what a single-view run returns).
        results['global'] = self._view_forward(ref_g, dist_g)

        if views == ['global']:
            return results if self.return_cache else results['global']

        B, _, H, W = ref.shape
        crop_size = int(min(H, W) * self.crop_ratio)

        if 'center' in views:
            top, left = (H - crop_size) // 2, (W - crop_size) // 2
            results['center'] = self._view_forward(ref[:, :, top:top + crop_size, left:left + crop_size],
                                                   dist[:, :, top:top + crop_size, left:left + crop_size])

        if 'texture' in views:
            # Cache mode stores a single texture crop (the strongest peak), like the hybrid cache.
            k = 1 if self.return_cache else self.texture_crops_k
            crop_scores = [self._view_forward(r, d) for r, d in self._texture_crops(ref, dist, crop_size, k)]
            if self.return_cache:
                results['texture'] = crop_scores[0]
            elif k == 1:
                results['texture'] = crop_scores[0]
            else:
                stacked = torch.stack(crop_scores, dim=0)  # (k, B)
                if self.texture_agg == 'mean':
                    results['texture'] = stacked.mean(dim=0)
                elif self.texture_agg == 'min':
                    results['texture'] = stacked.min(dim=0)[0]
                else:
                    raise ValueError(f"Unknown texture_agg: {self.texture_agg}")

        if self.return_cache:
            return results

        score = 0.0
        for v in views:
            score = score + self.view_weights.get(v, 0) * results[v]
        return score


def register():
    """Register into PyIQA (registration happens through the decorator above)."""
    pass


DEFAULT_CONFIGS['foundation_dino'] = {
    'metric_opts': {'type': 'FoundationDino'},
    'metric_mode': 'FR',
    'lower_better': False,
}
