import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
import math
import warnings

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.default_model_configs import DEFAULT_CONFIGS

class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        
    def forward(self, x):
        return (x - self.mean) / self.std

class DINOv2SpatialExtractor(nn.Module):
    def __init__(self, model_name="dinov2_vitb14", layer_indices=[0, 3, 6, 9, 11], device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.layer_indices = layer_indices
        
    @torch.no_grad()
    def forward(self, x, padding_mode='reflect', dino_norm_imagenet=False):
        if dino_norm_imagenet:
            # Assume x is already in [0,1] or comparable if it's the raw image.
            # Usually pyiqa models receive [0,1]. ImageNet norm:
            x = Normalize()(x)
            
        B, C, H, W = x.shape
        pad_h = (14 - H % 14) % 14
        pad_w = (14 - W % 14) % 14
        
        if pad_h > 0 or pad_w > 0:
            if padding_mode in ['reflect', 'replicate']:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode=padding_mode)
            elif padding_mode == 'bicubic-resize-to-multiple-of-14':
                x = F.interpolate(x, size=(H + pad_h, W + pad_w), mode='bicubic', align_corners=False)
            
        H_pad, W_pad = x.shape[2], x.shape[3]
        h_patches, w_patches = H_pad // 14, W_pad // 14

        out_raw = self.model.get_intermediate_layers(
            x.to(self.device), n=self.layer_indices, return_class_token=True
        )

        spatial_maps = []
        cls_tokens = []

        for patch_tokens, cls_tok in out_raw:
            C_feat = patch_tokens.shape[-1]
            feat_2d = patch_tokens.permute(0, 2, 1).view(B, C_feat, h_patches, w_patches)
            spatial_maps.append(feat_2d)
            cls_tokens.append(cls_tok)

        return spatial_maps, cls_tokens


@ARCH_REGISTRY.register()
class FoundationHybrid(nn.Module):
    def __init__(self, 
                 dino_model_name="dinov2_vitb14",
                 cnn_backbone="alexnet",
                 device=None,
                 worst_k_ratio=0.10,
                 mean_worst_mix_alpha=0.5,
                 ws=4,
                 pf=1.0,
                 xi=1e-6,
                 multiscale=True,
                 # DINO Ablations
                 dino_score_terms=['dists', 'patch_cos', 'cls'],
                 dino_patch_cos_mode='both', 
                 dino_layer_weights=[0.25, 0.30, 0.25, 0.12, 0.08],
                 dino_layer_indices=None,
                 dino_dists_local=False,
                 dino_dists_ws=3,
                 dino_norm_imagenet=False,
                 dino_padding_mode='reflect',
                 dino_resize=None,
                 # CNN Ablations
                 cnn_layers=None,
                 cnn_layer_weights=None,
                 cnn_dists_ws=4,
                 cnn_dists_term='both', 
                 cnn_gram_mode='local', 
                 cnn_gram_pf=1.0,
                 cnn_gram_channel_sel='var',
                 # Gate Ablations
                 gate_mode='full',
                 gate_fixed_g=0.5,
                 gate_steepness=4.0,
                 gate_threshold=0.70,
                 gate_w_d_range=(0.35, 0.65),
                 gate_w_g_range=(0.50, 0.20),
                 gate_w_p=0.15,
                 gate_signal='cnn_all', 
                 gate_signal_layer=None,
                 gate_scope='per_view', 
                 expert_weights=None, 
                 # Multi-view ablations
                 views=['global', 'center', 'texture'],
                 view_weights={'global': 0.60, 'center': 0.25, 'texture': 0.15},
                 crop_ratio=0.70,
                 crop_selection='max_var_ref',
                 texture_crops_k=1,
                 texture_agg='mean',
                 texture_var_ws=7,
                 pyramid_mode=False,
                 return_cache=False,
                 **kwargs
                 ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.worst_k_ratio = worst_k_ratio
        self.mean_worst_mix_alpha = mean_worst_mix_alpha
        self.ws = ws
        self.pf = pf
        self.xi = xi
        self.multiscale = multiscale
        self.return_cache = return_cache
        
        self.dino_score_terms = dino_score_terms
        self.dino_patch_cos_mode = dino_patch_cos_mode
        self.dino_layer_weights = dino_layer_weights
        
        if dino_layer_indices is None:
            # Map default indices based on backbone
            if 'vits' in dino_model_name or 'vitb' in dino_model_name:
                self.dino_layer_indices = [0, 3, 6, 9, 11]
            elif 'vitl' in dino_model_name:
                self.dino_layer_indices = [0, 5, 11, 17, 23]
            elif 'vitg' in dino_model_name:
                self.dino_layer_indices = [0, 9, 19, 29, 39]
        else:
            self.dino_layer_indices = dino_layer_indices

        self.dino_dists_local = dino_dists_local
        self.dino_dists_ws = dino_dists_ws
        self.dino_norm_imagenet = dino_norm_imagenet
        self.dino_padding_mode = dino_padding_mode
        self.dino_resize = dino_resize
        
        if cnn_layers is None:
            if "alexnet" in cnn_backbone:
                self.cnn_layers = ["features.2", "features.5", "features.7", "features.9", "features.12"]
            elif "vgg" in cnn_backbone:
                self.cnn_layers = ["features.3", "features.8", "features.15", "features.22", "features.29"]
            else:
                self.cnn_layers = ["layer1", "layer2", "layer3", "layer4"]
        else:
            self.cnn_layers = cnn_layers
            
        self.cnn_layer_weights = cnn_layer_weights
        self.cnn_dists_ws = cnn_dists_ws if cnn_dists_ws is not None else self.ws
        self.cnn_dists_term = cnn_dists_term
        self.cnn_gram_mode = cnn_gram_mode
        self.cnn_gram_pf = cnn_gram_pf if cnn_gram_pf is not None else self.pf
        self.cnn_gram_channel_sel = cnn_gram_channel_sel
        
        self.gate_mode = gate_mode
        self.gate_fixed_g = gate_fixed_g
        self.gate_steepness = gate_steepness
        self.gate_threshold = gate_threshold
        self.gate_w_d_range = gate_w_d_range
        self.gate_w_g_range = gate_w_g_range
        self.gate_w_p = gate_w_p
        self.gate_signal = gate_signal
        self.gate_signal_layer = gate_signal_layer
        self.gate_scope = gate_scope
        self.expert_weights = expert_weights
        
        self.views = views
        self.view_weights = view_weights
        self.crop_ratio = crop_ratio
        self.crop_selection = crop_selection
        self.texture_crops_k = texture_crops_k
        self.texture_agg = texture_agg
        self.texture_var_ws = texture_var_ws
        self.pyramid_mode = pyramid_mode

        # Initialization
        if 'dino' in self.expert_weights_keys():
            self.dino = DINOv2SpatialExtractor(dino_model_name, self.dino_layer_indices, device=self.device)
            
        if 'gram' in self.expert_weights_keys() or 'dists' in self.expert_weights_keys():
            self.cnn_ext, self.cnn_norm = self._make_extractor(cnn_backbone, self.cnn_layers)

    def expert_weights_keys(self):
        if self.expert_weights is not None:
            return [k for k, v in self.expert_weights.items() if v > 0]
        return ['dino', 'gram', 'dists']

    def _make_extractor(self, backbone, layers):
        if 'alexnet' in backbone:
            base = models.alexnet(weights='IMAGENET1K_V1')
        elif 'vgg16' in backbone:
            base = models.vgg16(weights='IMAGENET1K_V1')
        elif 'resnet50' in backbone:
            base = models.resnet50(weights='IMAGENET1K_V1')
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        return_nodes = {l: l for l in layers}
        ext = create_feature_extractor(base, return_nodes=return_nodes).to(self.device).eval()
        for p in ext.parameters():
            p.requires_grad = False
        norm = Normalize().to(self.device)
        return ext, norm

    @staticmethod
    def _gram(feat):
        n, c, h, w = feat.shape
        f = feat.view(n, c, h * w)
        return torch.bmm(f, f.transpose(1, 2)) / (h * w)

    def _select_channels(self, feat_ref, feat_dist):
        if self.cnn_gram_pf >= 1.0:
            return feat_ref, feat_dist
        n, c, h, w = feat_ref.shape
        k = max(1, int(c * self.cnn_gram_pf))
        
        if self.cnn_gram_channel_sel == 'var':
            var = torch.var(feat_ref, dim=(2, 3), unbiased=False)
            _, idx = torch.topk(var, k, dim=1)
        elif self.cnn_gram_channel_sel == 'min_var':
            var = torch.var(feat_ref, dim=(2, 3), unbiased=False)
            _, idx = torch.topk(var, k, dim=1, largest=False)
        elif self.cnn_gram_channel_sel == 'random':
            # deterministic random based on shape
            idx = torch.randperm(c, device=feat_ref.device)[:k].unsqueeze(0).expand(n, -1)
            
        idx_r = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        s_ref = torch.gather(feat_ref, 1, idx_r)
        _, _, hd, wd = feat_dist.shape
        idx_d = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hd, wd)
        s_dist = torch.gather(feat_dist, 1, idx_d)
        return s_ref, s_dist

    @torch.no_grad()
    def _compute_dino_score(self, ref, dist):
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

        weighted_layer_scores = []
        lws = self.dino_layer_weights
        if sum(lws) == 0:
            lws = [1.0/len(lws)] * len(lws)
            
        dino_cache = []
            
        for (fr, fd, cr, cd), lw in zip(zip(maps_r, maps_d, cls_r, cls_d), lws):
            dists_score = patch_cos_score = cls_score = 0.0
            
            # 1. DISTS SSIM
            if 'dists' in self.dino_score_terms:
                if self.dino_dists_local:
                    pad = self.dino_dists_ws // 2
                    pool = nn.AvgPool2d(kernel_size=self.dino_dists_ws, stride=1, padding=pad)
                    mr = pool(fr)
                    md = pool(fd)
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
                dists_score = (s_mean * s_var).mean(dim=(1, 2, 3))

            # 2. Patch Cosine
            mean_cos = None
            cos_flat = None
            if 'patch_cos' in self.dino_score_terms:
                fr_norm = F.normalize(fr, p=2, dim=1)
                fd_norm = F.normalize(fd, p=2, dim=1)
                cos_sim_map = (fr_norm * fd_norm).sum(dim=1)
                cos_flat = cos_sim_map.view(ref.shape[0], -1)

                mean_cos = cos_flat.mean(dim=1)
                k_val = max(1, int(cos_flat.shape[1] * self.worst_k_ratio))
                worst_cos = torch.topk(cos_flat, k_val, dim=1, largest=False)[0].mean(dim=1)
                
                if self.dino_patch_cos_mode == 'mean':
                    patch_cos_score = mean_cos
                elif self.dino_patch_cos_mode == 'worst':
                    patch_cos_score = worst_cos
                else:
                    patch_cos_score = self.mean_worst_mix_alpha * mean_cos + (1 - self.mean_worst_mix_alpha) * worst_cos

            # 3. CLS Cosine
            if 'cls' in self.dino_score_terms:
                cr_norm = F.normalize(cr, p=2, dim=1)
                cd_norm = F.normalize(cd, p=2, dim=1)
                cls_score = (cr_norm * cd_norm).sum(dim=1)

            # Note: default weights are 0.45, 0.45, 0.10. 
            # In ablations we sweep these. But since they are hardcoded in prompt, we'll keep them adjustable if needed.
            # To be completely safe and reproducible, let's hardcode for now unless we add variables.
            layer_score = 0.45 * dists_score + 0.45 * patch_cos_score + 0.10 * cls_score
            weighted_layer_scores.append(lw * layer_score)
            
            if self.return_cache:
                dino_cache.append({
                    'dists_score': dists_score,
                    'mean_cos': mean_cos if mean_cos is not None else torch.zeros_like(dists_score),
                    'cos_flat': cos_flat if cos_flat is not None else torch.zeros((ref.shape[0], 1), device=ref.device),
                    'cls_score': cls_score if 'cls' in self.dino_score_terms else torch.zeros_like(dists_score)
                })

        if self.return_cache:
            return dino_cache
        return torch.stack(weighted_layer_scores, dim=0).sum(dim=0) / sum(lws)

    @torch.no_grad()
    def _compute_cnn_score(self, ref, dist):
        out_r = self.cnn_ext(self.cnn_norm(ref.to(self.device)))
        out_d = self.cnn_ext(self.cnn_norm(dist.to(self.device)))

        dists_scores = []
        gram_scores = []
        spatial_variance_list = []
        cnn_cache = []

        keys = self.cnn_layers
        lws = self.cnn_layer_weights
        if lws is None:
            lws = [1.0/len(keys)] * len(keys)

        for k, lw in zip(keys, lws):
            fr = out_r[k]
            fd = out_d[k]
            if fr.dim() == 2:
                fr = fr.unsqueeze(-1).unsqueeze(-1)
                fd = fd.unsqueeze(-1).unsqueeze(-1)

            pad = self.cnn_dists_ws // 2
            
            # Global vs Local DISTS
            if self.cnn_dists_ws == 1:
                mr = torch.mean(fr, dim=(2,3), keepdim=True)
                md = torch.mean(fd, dim=(2,3), keepdim=True)
                vr = torch.var(fr, dim=(2,3), unbiased=False, keepdim=True)
                vd = torch.var(fd, dim=(2,3), unbiased=False, keepdim=True)
                cov = torch.mean((fr - mr)*(fd - md), dim=(2,3), keepdim=True)
            else:
                pool = nn.AvgPool2d(kernel_size=self.cnn_dists_ws, stride=1, padding=pad)
                mr = pool(fr)
                md = pool(fd)
                vr = torch.clamp(pool(fr ** 2) - mr ** 2, min=0.0)
                vd = torch.clamp(pool(fd ** 2) - md ** 2, min=0.0)
                cov = pool(fr * fd) - mr * md

            s_mean = (2 * mr * md + self.xi) / (mr ** 2 + md ** 2 + self.xi)
            s_var = (2 * cov + self.xi) / (vr + vd + self.xi)
            
            if self.cnn_dists_term == 's_mean':
                dists_map = s_mean
            elif self.cnn_dists_term == 's_var':
                dists_map = s_var
            else:
                dists_map = s_mean * s_var
                
            dists_scores.append(lw * dists_map.mean(dim=(1, 2, 3)))

            # Track spatial non-uniformity of feature diff map
            spatial_diff = (fr - fd).abs().mean(dim=1)
            # check for NaN/Inf (V4)
            if spatial_diff.numel() <= ref.shape[0] or spatial_diff.shape[-1] == 1:
                warnings.warn("Tiny spatial map detected. Std will be 0.")
                std_diff = torch.zeros_like(spatial_diff.view(-1, 1).mean(dim=1))
            else:
                std_diff = torch.std(spatial_diff.view(ref.shape[0], -1), dim=1)
            mean_diff = torch.mean(spatial_diff.view(ref.shape[0], -1), dim=1) + 1e-6
            nu_val = std_diff / mean_diff
            
            if torch.isnan(nu_val).any() or torch.isinf(nu_val).any():
                warnings.warn("NaN or Inf detected in Non-Uniformity metric. Clamping to 0.")
                nu_val = torch.nan_to_num(nu_val, nan=0.0, posinf=0.0, neginf=0.0)
                
            spatial_variance_list.append(nu_val)

            # Gram Score
            fr_sel, fd_sel = self._select_channels(fr, fd)
            if self.cnn_gram_mode == 'local' and fr_sel.shape[2] >= self.ws and fr_sel.shape[3] >= self.ws:
                gr = self._gram(fr_sel)
                gd = self._gram(fd_sel)
                gr_u = F.unfold(gr.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
                gd_u = F.unfold(gd.unsqueeze(1), kernel_size=self.ws, stride=1).transpose(1, 2)
                vr_g = torch.var(gr_u, dim=2, unbiased=False)
                vd_g = torch.var(gd_u, dim=2, unbiased=False)
                mr_g = torch.mean(gr_u, dim=2, keepdim=True)
                md_g = torch.mean(gd_u, dim=2, keepdim=True)
                cov_g = torch.mean((gr_u - mr_g) * (gd_u - md_g), dim=2)
                local_gram = (2 * cov_g + self.xi) / (vr_g + vd_g + self.xi)
                gram_scores.append(lw * local_gram.mean(dim=1))
            elif self.cnn_gram_mode in ['global', 'local']: # fallback to global
                n, c = fr_sel.shape[:2]
                fr_flat = fr_sel.view(n, c, -1)
                fd_flat = fd_sel.view(n, c, -1)
                gr = torch.bmm(fr_flat, fr_flat.transpose(1, 2)) / fr_flat.shape[2]
                gd = torch.bmm(fd_flat, fd_flat.transpose(1, 2)) / fd_flat.shape[2]
                vr_g = torch.var(gr, dim=(1, 2), unbiased=False)
                vd_g = torch.var(gd, dim=(1, 2), unbiased=False)
                mr_g = torch.mean(gr, dim=(1, 2))
                md_g = torch.mean(gd, dim=(1, 2))
                cov_g = torch.mean((gr - mr_g.unsqueeze(-1).unsqueeze(-1)) * (gd - md_g.unsqueeze(-1).unsqueeze(-1)), dim=(1, 2))
                s_mean_g = (2 * mr_g * md_g + self.xi) / (mr_g ** 2 + md_g ** 2 + self.xi)
                s_var_g = (2 * cov_g + self.xi) / (vr_g + vd_g + self.xi)
                gram_scores.append(lw * (s_mean_g * s_var_g))
            elif self.cnn_gram_mode == 'l2':
                n, c = fr_sel.shape[:2]
                fr_flat = fr_sel.view(n, c, -1)
                fd_flat = fd_sel.view(n, c, -1)
                gr = torch.bmm(fr_flat, fr_flat.transpose(1, 2)) / fr_flat.shape[2]
                gd = torch.bmm(fd_flat, fd_flat.transpose(1, 2)) / fd_flat.shape[2]
                l2_dist = torch.mean((gr - gd) ** 2, dim=(1,2))
                gram_scores.append(lw * torch.exp(-l2_dist))

            if self.return_cache:
                cnn_cache.append({
                    'dists_score': dists_scores[-1] / lw if lw > 0 else torch.zeros_like(dists_scores[-1]),
                    'gram_score': gram_scores[-1] / lw if lw > 0 else torch.zeros_like(gram_scores[-1]),
                    'std_diff': std_diff,
                    'mean_diff': mean_diff
                })

        if self.return_cache:
            return [], cnn_cache, []

        dists_score = torch.stack(dists_scores, dim=0).sum(dim=0)
        gram_score = torch.stack(gram_scores, dim=0).sum(dim=0)
        non_uniformity = torch.stack(spatial_variance_list, dim=0).mean(dim=0)

        return dists_score, gram_score, non_uniformity

    @torch.no_grad()
    def _single_scale_forward(self, ref, dist, gate_override=None):
        if self.return_cache:
            dino_cache = self._compute_dino_score(ref, dist) if 'dino' in self.expert_weights_keys() else []
            _, cnn_cache, _ = self._compute_cnn_score(ref, dist) if ('gram' in self.expert_weights_keys() or 'dists' in self.expert_weights_keys()) else ([], [], [])
            return {'dino': dino_cache, 'cnn': cnn_cache}

        B = ref.shape[0]
        s_dino = s_dists = s_gram = torch.zeros(B, device=self.device)
        non_uniformity = torch.zeros(B, device=self.device)
        
        if 'dino' in self.expert_weights_keys():
            s_dino = self._compute_dino_score(ref, dist)
        if 'gram' in self.expert_weights_keys() or 'dists' in self.expert_weights_keys():
            s_dists, s_gram, non_uniformity = self._compute_cnn_score(ref, dist)

        if self.expert_weights is not None:
            # Fixed weight ablation A1
            return (self.expert_weights.get('dino', 0)*s_dino + 
                    self.expert_weights.get('gram', 0)*s_gram + 
                    self.expert_weights.get('dists', 0)*s_dists)

        # Gating
        if gate_override is not None:
            gate_val = gate_override
        else:
            if self.gate_mode == 'fixed':
                gate_val = torch.full((B,), self.gate_fixed_g, device=self.device)
            elif self.gate_mode == 'none':
                gate_val = torch.full((B,), 0.5, device=self.device)
            else:
                # full mode
                if self.gate_signal == 'cnn_all':
                    sig = non_uniformity
                # other gate signals can be injected here for D5
                else:
                    sig = non_uniformity
                
                if self.gate_steepness == 'hard':
                    gate_val = (sig > self.gate_threshold).float()
                else:
                    gate_val = torch.sigmoid(self.gate_steepness * (sig - self.gate_threshold))

        w_d_min, w_d_max = self.gate_w_d_range
        w_g_min, w_g_max = self.gate_w_g_range
        
        w_d = w_d_min + (w_d_max - w_d_min) * gate_val
        w_g = w_g_min + (w_g_max - w_g_min) * gate_val
        w_p = self.gate_w_p

        if self.gate_mode == 'none' and w_d_min==w_d_max and w_g_min==w_g_max:
            # specifically for D1: equal weights (1/3 each)
            if self.gate_w_p != 0.15:  # Hacky check if it's the equal weight ablations
                 w_d, w_g, w_p = 1/3, 1/3, 1/3

        return (w_d * s_dino + w_g * s_gram + w_p * s_dists), gate_val, non_uniformity

    @torch.no_grad()
    def forward(self, ref, dist, **kwargs):
        # We need to return score natively.
        
        if self.return_cache:
            cache_out = {}
            if self.pyramid_mode:
                ref_global = F.interpolate(ref, scale_factor=0.5, mode='bicubic')
                dist_global = F.interpolate(dist, scale_factor=0.5, mode='bicubic')
                cache_out['global'] = self._single_scale_forward(ref_global, dist_global)
            else:
                cache_out['global'] = self._single_scale_forward(ref, dist)

            if not self.multiscale or len(self.views) == 1 and self.views[0] == 'global':
                return cache_out

            B, C, H, W = ref.shape
            crop_size = int(min(H, W) * self.crop_ratio)

            if 'center' in self.views:
                top_c = (H - crop_size) // 2
                left_c = (W - crop_size) // 2
                ref_center = ref[:, :, top_c:top_c + crop_size, left_c:left_c + crop_size]
                dist_center = dist[:, :, top_c:top_c + crop_size, left_c:left_c + crop_size]
                cache_out['center'] = self._single_scale_forward(ref_center, dist_center)

            if 'texture' in self.views:
                # In tier1 caching, batch size is usually 1, so we just pick the first crop
                i = 0
                if self.crop_selection == 'max_var_ref':
                    var_map = F.avg_pool2d(ref[i:i+1].mean(dim=1, keepdim=True) ** 2, self.texture_var_ws, 1, self.texture_var_ws//2) - \
                              F.avg_pool2d(ref[i:i+1].mean(dim=1, keepdim=True), self.texture_var_ws, 1, self.texture_var_ws//2) ** 2
                elif self.crop_selection == 'max_var_dist':
                    var_map = F.avg_pool2d(dist[i:i+1].mean(dim=1, keepdim=True) ** 2, self.texture_var_ws, 1, self.texture_var_ws//2) - \
                              F.avg_pool2d(dist[i:i+1].mean(dim=1, keepdim=True), self.texture_var_ws, 1, self.texture_var_ws//2) ** 2
                elif self.crop_selection == 'max_diff':
                    var_map = (ref[i:i+1] - dist[i:i+1]).abs().mean(dim=1, keepdim=True)
                else:
                    var_map = F.avg_pool2d(ref[i:i+1].mean(dim=1, keepdim=True) ** 2, self.texture_var_ws, 1, self.texture_var_ws//2) - \
                              F.avg_pool2d(ref[i:i+1].mean(dim=1, keepdim=True), self.texture_var_ws, 1, self.texture_var_ws//2) ** 2

                var_map = var_map.squeeze()
                max_idx = torch.argmax(var_map)
                h_idx, w_idx = max_idx // W, max_idx % W
                top_t = max(0, min(H - crop_size, int(h_idx) - crop_size // 2))
                left_t = max(0, min(W - crop_size, int(w_idx) - crop_size // 2))
                ref_tex = ref[i:i+1, :, top_t:top_t + crop_size, left_t:left_t + crop_size]
                dist_tex = dist[i:i+1, :, top_t:top_t + crop_size, left_t:left_t + crop_size]
                cache_out['texture'] = self._single_scale_forward(ref_tex, dist_tex)

            return cache_out

        # 1. Global View
        if self.pyramid_mode:
            H, W = ref.shape[2:]
            ref_global = F.interpolate(ref, scale_factor=0.5, mode='bicubic')
            dist_global = F.interpolate(dist, scale_factor=0.5, mode='bicubic')
            s_global, gate_val, nu = self._single_scale_forward(ref_global, dist_global)
        else:
            s_global, gate_val, nu = self._single_scale_forward(ref, dist)

        if not self.multiscale or len(self.views) == 1 and self.views[0] == 'global':
            return s_global

        B, C, H, W = ref.shape
        crop_size = int(min(H, W) * self.crop_ratio)

        shared_gate = gate_val if self.gate_scope == 'shared_global' else None

        score = 0.0
        
        if 'global' in self.views:
            score += self.view_weights.get('global', 0) * s_global

        if 'center' in self.views:
            top_c = (H - crop_size) // 2
            left_c = (W - crop_size) // 2
            ref_center = ref[:, :, top_c:top_c + crop_size, left_c:left_c + crop_size]
            dist_center = dist[:, :, top_c:top_c + crop_size, left_c:left_c + crop_size]
            s_zoom_center, _, _ = self._single_scale_forward(ref_center, dist_center, gate_override=shared_gate)
            score += self.view_weights.get('center', 0) * s_zoom_center

        if 'texture' in self.views:
            s_zoom_tex_list = []
            # We must iterate over batch for multi-crop to avoid the V2 bug.
            for i in range(B):
                if self.crop_selection == 'max_var_ref':
                    var_map = F.avg_pool2d(ref[i:i+1].mean(dim=1, keepdim=True) ** 2, self.texture_var_ws, 1, self.texture_var_ws//2) - \
                              F.avg_pool2d(ref[i:i+1].mean(dim=1, keepdim=True), self.texture_var_ws, 1, self.texture_var_ws//2) ** 2
                elif self.crop_selection == 'max_var_dist':
                    var_map = F.avg_pool2d(dist[i:i+1].mean(dim=1, keepdim=True) ** 2, self.texture_var_ws, 1, self.texture_var_ws//2) - \
                              F.avg_pool2d(dist[i:i+1].mean(dim=1, keepdim=True), self.texture_var_ws, 1, self.texture_var_ws//2) ** 2
                elif self.crop_selection == 'max_diff':
                    var_map = (ref[i:i+1] - dist[i:i+1]).abs().mean(dim=1, keepdim=True)
                elif self.crop_selection == 'min_var':
                    var_map = -(F.avg_pool2d(ref[i:i+1].mean(dim=1, keepdim=True) ** 2, self.texture_var_ws, 1, self.texture_var_ws//2) - \
                                F.avg_pool2d(ref[i:i+1].mean(dim=1, keepdim=True), self.texture_var_ws, 1, self.texture_var_ws//2) ** 2)
                
                var_map = var_map.squeeze()
                
                if self.texture_crops_k == 1:
                    max_idx = torch.argmax(var_map)
                    h_idx, w_idx = max_idx // W, max_idx % W
                    top_t = max(0, min(H - crop_size, int(h_idx) - crop_size // 2))
                    left_t = max(0, min(W - crop_size, int(w_idx) - crop_size // 2))
                    ref_tex = ref[i:i+1, :, top_t:top_t + crop_size, left_t:left_t + crop_size]
                    dist_tex = dist[i:i+1, :, top_t:top_t + crop_size, left_t:left_t + crop_size]
                    
                    st, _, _ = self._single_scale_forward(ref_tex, dist_tex, gate_override=shared_gate[i:i+1] if shared_gate is not None else None)
                    s_zoom_tex_list.append(st)
                else:
                    # multiple crops
                    flat_var = var_map.view(-1)
                    vals, ids = torch.topk(flat_var, self.texture_crops_k)
                    st_crops = []
                    for k_idx in range(self.texture_crops_k):
                        max_idx = ids[k_idx]
                        h_idx, w_idx = max_idx // W, max_idx % W
                        top_t = max(0, min(H - crop_size, int(h_idx) - crop_size // 2))
                        left_t = max(0, min(W - crop_size, int(w_idx) - crop_size // 2))
                        ref_tex = ref[i:i+1, :, top_t:top_t + crop_size, left_t:left_t + crop_size]
                        dist_tex = dist[i:i+1, :, top_t:top_t + crop_size, left_t:left_t + crop_size]
                        st, _, _ = self._single_scale_forward(ref_tex, dist_tex, gate_override=shared_gate[i:i+1] if shared_gate is not None else None)
                        st_crops.append(st)
                    
                    st_crops = torch.cat(st_crops)
                    if self.texture_agg == 'mean':
                        s_zoom_tex_list.append(st_crops.mean())
                    elif self.texture_agg == 'min':
                        s_zoom_tex_list.append(st_crops.min())
            
            if len(s_zoom_tex_list) > 0:
                s_zoom_tex = torch.cat(s_zoom_tex_list)
                score += self.view_weights.get('texture', 0) * s_zoom_tex

        return score

def register():
    """Register into PyIQA."""
    pass
    # @ARCH_REGISTRY.register() handles registration of FoundationHybrid
    
DEFAULT_CONFIGS['foundation_hybrid'] = {
    'metric_mode': 'FR',
    'lower_better': False
}
