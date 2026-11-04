import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import List, Dict, Optional, Tuple
from timm.layers import trunc_normal_


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))
        self.bias   = nn.Parameter(torch.zeros(num_channels, 1, 1))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var  = x.var (dim=(1, 2, 3), keepdim=True, unbiased=False)
        x    = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight + self.bias


class SpatialReduction(nn.Module):
    def __init__(self, channels: int, sr_ratio: int = 8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=sr_ratio, stride=sr_ratio, padding=0, bias=False),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

class BidirectionSpatialCrossAttention(nn.Module):
    def __init__(self, channels: int, head_dim: int = 4, channel_reduction_ratio: int = 2, sr_ratio: int = 2):
        super().__init__()
        
        self.head_dim  = head_dim
        self.inner_dim = channels // channel_reduction_ratio
        assert self.inner_dim % self.head_dim == 0
        self.heads     = self.inner_dim//self.head_dim
        self.scale     = self.head_dim ** -0.5

        self.sr       = SpatialReduction(channels, sr_ratio=sr_ratio)
        self.norm_q   = nn.LayerNorm(channels)
        self.norm_kv  = nn.LayerNorm(channels)
        self.q_proj   = nn.Linear(channels, self.inner_dim, bias=False)
        self.k_proj   = nn.Linear(channels, self.inner_dim, bias=False)
        self.v_proj   = nn.Linear(channels, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, channels, bias=False)

    def _get_2d_sinusoid(self, H: int, W: int, C: int, device: torch.device) -> torch.Tensor:
        key = f"pe_{H}_{W}_{C}"

        if not hasattr(self, key):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij'
            )
            feat_dim = C // 4
            omega = torch.exp(
                torch.arange(feat_dim, device=device).float() * -(math.log(10000.0) / feat_dim)
            )
            
            sin_x = torch.sin(grid_x.unsqueeze(-1) * omega)
            cos_x = torch.cos(grid_x.unsqueeze(-1) * omega)
            sin_y = torch.sin(grid_y.unsqueeze(-1) * omega)
            cos_y = torch.cos(grid_y.unsqueeze(-1) * omega)
            
            emb = torch.cat([sin_x, cos_x, sin_y, cos_y], dim=-1)
            
            if emb.shape[-1] < C:
                emb = F.pad(emb, (0, C - emb.shape[-1]))
            emb = emb.permute(2, 0, 1).unsqueeze(0)
            self.register_buffer(key, emb, persistent=False)
            
        return getattr(self, key)

    @staticmethod
    def _to_seq(x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, _, N, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, self.inner_dim)

    def _cross_attend(self, query_feat: torch.Tensor, kv_reduced: torch.Tensor) -> torch.Tensor:
        B, C, H, W     = query_feat.shape
        _, _, H_r, W_r = kv_reduced.shape
        pe_q  = self._get_2d_sinusoid(H,   W,   C, query_feat.device)
        pe_kv = self._get_2d_sinusoid(H_r, W_r, C, kv_reduced.device)
        q_seq  = self.norm_q (self._to_seq(query_feat + pe_q))
        kv_seq = self.norm_kv(self._to_seq(kv_reduced + pe_kv))
        Q = self._split_heads(self.q_proj(q_seq))
        K = self._split_heads(self.k_proj(kv_seq))
        V = self._split_heads(self.v_proj(kv_seq))
        out = F.scaled_dot_product_attention(Q, K, V, scale=self.scale)
        out = self.out_proj(self._merge_heads(out))
        return out.transpose(1, 2).view(B, C, H, W)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a_reduced = self.sr(a)
        b_reduced = self.sr(b)
        return self._cross_attend(a, b_reduced), self._cross_attend(b, a_reduced)

class TTTCrossAttn(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        head_dim = dim // num_heads
        self.dim = dim
        self.num_heads = num_heads

        self.q_proj = nn.Linear(dim, dim + head_dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2 + head_dim * 2, bias=qkv_bias)
        self.w1 = nn.Parameter(torch.zeros(1, self.num_heads, head_dim, head_dim))
        self.w2 = nn.Parameter(torch.zeros(1, self.num_heads, head_dim, head_dim))
        self.w3 = nn.Parameter(torch.zeros(head_dim, 1, 3, 3))
        trunc_normal_(self.w1, std=.02)
        trunc_normal_(self.w2, std=.02)
        trunc_normal_(self.w3, std=.02)
        self.proj = nn.Linear(dim + head_dim, dim)

        equivalent_head_dim = 9
        self.scale = equivalent_head_dim ** -0.5

        # fast-weight probe (analysis only; off by default)
        self.probe = False
        self.direction = ''
        self.last_probe = []

    def inner_train_simplified_swiglu(self, k, v, w1, w2, lr=1.0):
        z1 = k @ w1
        z2 = k @ w2
        sig = F.sigmoid(z2)
        a = z2 * sig
        e = - v / float(v.shape[2]) * self.scale
        g1 = k.transpose(-2, -1) @ (e * a)
        g2 = k.transpose(-2, -1) @ (e * z1 * (sig * (1.0 + z2 * (1.0 - sig))))
        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)
        w1, w2 = w1 - lr * g1, w2 - lr * g2
        return w1, w2

    def inner_train_3x3dwc(self, k, v, w, lr=1.0, implementation='prod'):
        B, C, H, W = k.shape
        e = - v / float(v.shape[2] * v.shape[3]) * self.scale
        if implementation == 'conv':
            g = F.conv2d(k.reshape(1, B * C, H, W), e.reshape(B * C, 1, H, W), padding=1, groups=B * C)
            g = g.transpose(0, 1)
        elif implementation == 'prod':
            k = F.pad(k, (1, 1, 1, 1))
            outs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ys = 1 + dy
                    xs = 1 + dx
                    dot = (k[:, :, ys: ys + H, xs: xs + W] * e).sum(dim=(-2, -1))
                    outs.append(dot)
            g = torch.stack(outs, dim=-1).reshape(B * C, 1, 3, 3)
        else:
            raise NotImplementedError
        g = g / (g.norm(dim=[-2, -1], keepdim=True) + 1.0)
        w = w.repeat(B, 1, 1, 1) - lr * g
        return w

    def forward(self, x_q, x_kv, hq, wq, hkv, wkv, rope_q=None, rope_kv=None):
        b, nq, c = x_q.shape
        _, nkv, _ = x_kv.shape
        d = c // self.num_heads

        q1, q2 = torch.split(self.q_proj(x_q), [c, d], dim=-1)
        k1, v1, k2, v2 = torch.split(self.kv_proj(x_kv), [c, c, d, d], dim=-1)

        if rope_q is not None:
            q1 = rope_q(q1.reshape(b, hq, wq, c)).reshape(b, nq, self.num_heads, d).transpose(1, 2)
        else:
            q1 = q1.reshape(b, nq, self.num_heads, d).transpose(1, 2)
        if rope_kv is not None:
            k1 = rope_kv(k1.reshape(b, hkv, wkv, c)).reshape(b, nkv, self.num_heads, d).transpose(1, 2)
        else:
            k1 = k1.reshape(b, nkv, self.num_heads, d).transpose(1, 2)
        v1 = v1.reshape(b, nkv, self.num_heads, d).transpose(1, 2)

        q2 = q2.reshape(b, hq, wq, d).permute(0, 3, 1, 2)
        k2 = k2.reshape(b, hkv, wkv, d).permute(0, 3, 1, 2)
        v2 = v2.reshape(b, hkv, wkv, d).permute(0, 3, 1, 2)

        w1, w2 = self.inner_train_simplified_swiglu(k1, v1, self.w1, self.w2)
        w3 = self.inner_train_3x3dwc(k2, v2, self.w3, implementation='prod')

        if self.probe:
            with torch.no_grad():
                d_lin = ((w1 - self.w1).flatten(1).norm(dim=1) + (w2 - self.w2).flatten(1).norm(dim=1))
                d_spa = (w3.reshape(b, d, 9) - self.w3.reshape(1, d, 9)).flatten(1).norm(dim=1)
                self.last_probe.append(dict(direction=self.direction, d_linear=d_lin.float().cpu(),
                                            d_spatial=d_spa.float().cpu()))

        x1 = (q1 @ w1) * F.silu(q1 @ w2)
        x1 = x1.transpose(1, 2).reshape(b, nq, c)
        x2 = F.conv2d(q2.reshape(1, b * d, hq, wq), w3, padding=1, groups=b * d)
        x2 = x2.reshape(b, d, nq).transpose(1, 2)

        x = torch.cat([x1, x2], dim=-1)
        x = self.proj(x)
        return x

class BidirectionTTTCrossAttn(nn.Module):
    def __init__(self, channels: int, head_dim: int = 32, **kwargs):
        super().__init__()
        assert channels % head_dim == 0
        num_heads = channels // head_dim
        self.ttt = TTTCrossAttn(channels, num_heads)

    @staticmethod
    def _to_seq(x):
        return x.flatten(2).transpose(1, 2)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        B, C, Ha, Wa = a.shape
        _, _, Hb, Wb = b.shape

        # `a` is queried, `b` supplies the keys/values the fast weights are fitted on.
        # In HedgeFormer, a = MS stream and b = DEM stream, so the first call is  DEM -> MS.
        self.ttt.direction = 'dem->ms'
        ctx_from_post = self.ttt(self._to_seq(a), self._to_seq(b), Ha, Wa, Hb, Wb)
        self.ttt.direction = 'ms->dem'
        ctx_from_pre  = self.ttt(self._to_seq(b), self._to_seq(a), Hb, Wb, Ha, Wa)

        ctx_from_post = ctx_from_post.transpose(1, 2).reshape(B, C, Ha, Wa)
        ctx_from_pre  = ctx_from_pre.transpose(1, 2).reshape(B, C, Hb, Wb)
        return ctx_from_post, ctx_from_pre

class _ConvMixer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mixer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(x)


class MLP_F(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, in_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MLPMixer(nn.Module):
    def __init__(
        self, 
        channels: int, 
        height: int, 
        width: int, 
        tokens_hidden_dim: int, 
        channels_hidden_dim: int
    ):
        super().__init__()
        self.num_tokens = height * width

        self.fusion_proj = nn.Conv2d(channels * 2, channels, kernel_size=1)

        self.norm1 = nn.LayerNorm(channels)
        self.token_mixer = MLP_F(in_features=self.num_tokens, hidden_features=tokens_hidden_dim)

        self.norm2 = nn.LayerNorm(channels)
        self.channel_mixer = MLP_F(in_features=channels, hidden_features=channels_hidden_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

        B0, _, H0, W0 = x1.shape
        if H0 * W0 != self.num_tokens:
            raise ValueError(f'token mixer built for {self.num_tokens} tokens but got {H0}x{W0}={H0*W0}; '
                             'the fusion variants SRA/TTT need input size == img_size given at construction (default 256)')
        x = torch.cat([x1, x2], dim=1) 
        x = self.fusion_proj(x)
        
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        norm_x = self.norm1(x)
        norm_x = norm_x.transpose(1, 2)
        mixed_tokens = self.token_mixer(norm_x)
        x = x + mixed_tokens.transpose(1, 2)

        norm_x = self.norm2(x)

        mixed_channels = self.channel_mixer(norm_x)
        x = x + mixed_channels
        x = x.transpose(1, 2).unflatten(2, (H, W))
        return x


class FusionEncoder(nn.Module):
    def __init__(self, channels: int,
        head_dim:                int = 32,
        sr_ratio:                int = 4,
        channel_reduction_ratio: int = 2,
        attention_type:          str = 'TTT',
        height:                  int = 64,
        width:                   int = 64,
        tokens_hidden_dim:       int = 256):
        super().__init__()
        self.attention_type = attention_type
        if self.attention_type == 'TTT':
            self.cross_attn = BidirectionTTTCrossAttn(channels, head_dim=head_dim)
            self.norm_post    = LayerNorm2d(channels)
            self.norm_pre     = LayerNorm2d(channels)
            self.fusion_mixer = (_MLPMixer(channels, height, width, tokens_hidden_dim, channels * 4)
                                 if FUSION_MIXER == 'token' else _ConvMixer(channels * 2, channels))
        elif self.attention_type == 'SRA':
            self.cross_attn   = BidirectionSpatialCrossAttention(channels, head_dim=head_dim, sr_ratio=sr_ratio, channel_reduction_ratio=channel_reduction_ratio)
            self.norm_post    = LayerNorm2d(channels)
            self.norm_pre     = LayerNorm2d(channels)
            self.fusion_mixer = (_MLPMixer(channels, height, width, tokens_hidden_dim, channels * 4)
                                 if FUSION_MIXER == 'token' else _ConvMixer(channels * 2, channels))
        else:                                            # 'Simple': concat + conv mixer (NOT the token mixer)
            self.fusion_mixer = _ConvMixer(channels * 2, channels)
            
    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        if self.attention_type == 'TTT' or self.attention_type == 'SRA':
            ctx_from_post, ctx_from_pre = self.cross_attn(post, pre)
            post_prime = self.norm_post(post + ctx_from_post)
            pre_prime  = self.norm_pre (pre  + ctx_from_pre)
            if isinstance(self.fusion_mixer, _MLPMixer):        # token mixer concatenates its two inputs itself
                return self.fusion_mixer(post_prime, pre_prime)
            return self.fusion_mixer(torch.cat([post_prime, pre_prime], dim=1))
        else:
            return self.fusion_mixer(torch.cat([pre, post], dim=1))


class _Head(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 hidden_dim: Optional[int] = None):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      bias=False, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class MLP(nn.Module):
    def __init__(self, skip_channels: int, segmentation_channels: int):
        super().__init__()
        self.linear = nn.Linear(skip_channels, segmentation_channels)

    def forward(self, x: torch.Tensor):
        batch, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.linear(x)
        x = x.transpose(1, 2).reshape(batch, -1, height, width)
        return x

class MLP_Decoder(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        decoder_dim: int = 256,
    ):
        super().__init__()
        if in_channels[1] == 0:
            in_channels = [
                channel for index, channel in enumerate(in_channels) if index != 1
            ]
        in_channels = in_channels[::-1]
        self.mlp_stage = nn.ModuleList(
            [MLP(channel, decoder_dim) for channel in in_channels[:-1]]
        )
        self.fuse_stage = nn.Sequential(
            nn.Conv2d(
                (len(in_channels) - 1) * decoder_dim,
                decoder_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        size = [dim.shape[-1] for dim in features]
        target_size = max(size)
        
        features = features[2:] if features[1].size(1) == 0 else features[1:]
        features = features[::-1]

        resized_features = []
        for mlp_layer, feature in zip(self.mlp_stage, features):
            projected = mlp_layer(feature)
            resized = F.interpolate(
                projected, size=target_size, mode="bilinear", align_corners=False
            )
            resized_features.append(resized)

        return self.fuse_stage(torch.cat(resized_features, dim=1))

class SegFormerDecoder(nn.Module):
    """Old decoder (3 stages, 1x1 conv + BN + ReLU per stage, bilinear upsample to the first stage, concat, fuse).
    Drop-in for MLP_Decoder: same constructor (raw encoder channel list) and same forward (raw feature list)."""
    def __init__(self, in_channels: List[int], decoder_dim: int = 128):
        super().__init__()
        if in_channels[1] == 0:                       # raw smp list: [input, dummy(0), stage1, stage2, stage3]
            in_channels = in_channels[2:]
        self.mlp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, decoder_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(inplace=True),
            )
            for c in in_channels
        ])
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(decoder_dim * len(in_channels), decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        if features[1].size(1) == 0:                  # raw list -> keep only the stage features
            features = features[2:]
        target_size = features[0].shape[-2:]
        projected = []
        for mlp, feat in zip(self.mlp_layers, features):
            p = mlp(feat)
            p = F.interpolate(p, size=target_size, mode='bilinear', align_corners=False)
            projected.append(p)
        return self.fusion_conv(torch.cat(projected, dim=1))


def _make_decoder(in_channels, decoder_dim):
    if DECODER == 'segformer':
        return SegFormerDecoder(in_channels, decoder_dim)
    if DECODER == 'mlp':
        return MLP_Decoder(in_channels, decoder_dim)
    raise ValueError(f"DECODER must be 'segformer' or 'mlp', got {DECODER!r}")


# Mixer used by SRA / TTT after cross-attention ('Simple' always uses the conv mixer):
#   'conv'  = old conv mixer (any input size, keys identical to your old checkpoints)
#   'token' = token/channel MLP mixer (input size fixed to img_size, default 256)
FUSION_MIXER = 'conv'

# Decoder used by EVERY variant:
#   'segformer' = conv decoder (1x1 conv + BN + ReLU per stage, works at stride 4)
#   'mlp'       = MLP_Decoder (linear projections, no conv, fuses at full input resolution)
# Checkpoints only load with the same setting they were trained with (the key names differ).
DECODER = 'segformer'

DEM_CH, MS_CH = 1, 6
ENCODER_NAME = os.environ.get('HEDGE_ENCODER', 'mit_b4')   # 'mit_b0' is used only by the CPU smoke test

# variant name -> constructor kwargs.  'Simple' = concat + conv mixer at each of the 3 scales.
VARIANTS = {
    'Simple':   dict(attention_type='Simple', modality='both'),
    'SRA':      dict(attention_type='SRA',    modality='both'),
    'TTT':      dict(attention_type='TTT',    modality='both'),
    'Early':    dict(attention_type='Early',  modality='both'),   # 7-channel input, one encoder
    'Late':     dict(attention_type='Late',   modality='both'),   # two full branches, logits averaged
    'DEM_only': dict(attention_type='Simple', modality='dem'),
    'MS_only':  dict(attention_type='Simple', modality='ms'),
}


def _stage_feats(feats, in_ch):
    """Drop the identity feature (in_ch channels) and smp's zero-channel dummy."""
    return [f for f in feats if f.shape[1] > in_ch]


def _stage_channels(encoder, in_ch):
    return [c for c in encoder.out_channels if c > in_ch]


class HedgeFormer(nn.Module):
    """
    forward(dtm, ms) -> {'seg_logits': (B, num_class, H, W)}     (same signature for EVERY variant;
    single-modality variants simply ignore the tensor they do not use, so main/validation need no branching).

    modality='both' + attention_type in {'Simple','SRA','TTT'} : 3-scale mid-level fusion.
        Simple -> concat + conv mixer;  SRA / TTT -> cross-attention + token/channel MLP mixer (fixed to img_size)
    modality='both' + 'Early' : concat(ms, dtm) -> one 7-channel MiT-B5 -> SegFormer decoder
    modality='both' + 'Late'  : two independent encoder+decoder+head branches, logits averaged
    modality='dem' | 'ms'     : one encoder, the SAME SegFormerDecoder/_Head as the dual model
    """

    def __init__(
        self,
        num_class:               int = 1,
        head_dim:                Optional[int] = None,     # None -> 32 for MiT-B5, 16 for the tiny smoke-test encoder
        sr_ratio:                int = 4,
        channel_reduction_ratio: int = 2,
        decoder_dim:             int = 128,
        attention_type:          str = 'TTT',
        modality:                str = 'both',
        encoder_weights:         Optional[str] = 'imagenet',
        img_size:                int = 256,      # token mixer of SRA/TTT is tied to this input size
        tokens_hidden_dim:       int = 256,
    ):
        super().__init__()
        head_dim = head_dim or (16 if ENCODER_NAME == 'mit_b0' else 32)
        assert modality in ('both', 'dem', 'ms'), modality
        assert attention_type in ('Simple', 'SRA', 'TTT', 'Early', 'Late'), attention_type
        self.modality = modality
        self.attention_type = attention_type
        self.mode = modality if modality != 'both' else (
            attention_type.lower() if attention_type in ('Early', 'Late') else 'mid')
        w = encoder_weights

        def enc(in_ch):
            return smp.encoders.get_encoder(ENCODER_NAME, in_channels=in_ch, depth=4, weights=w)

        if self.mode == 'early':
            self.encoder_early = enc(DEM_CH + MS_CH)
            self.decoder = _make_decoder(list(self.encoder_early.out_channels), decoder_dim)
            self.seg_head = _Head(decoder_dim, num_class)
        elif self.mode == 'late':
            self.encoder_dem = enc(DEM_CH)
            self.encoder_ms = enc(MS_CH)
            self.decoder_dem = _make_decoder(list(self.encoder_dem.out_channels), decoder_dim)
            self.decoder_ms = _make_decoder(list(self.encoder_ms.out_channels), decoder_dim)
            self.seg_head_dem = _Head(decoder_dim, num_class)
            self.seg_head_ms = _Head(decoder_dim, num_class)
        elif self.mode == 'dem':
            self.encoder_dem = enc(DEM_CH)
            self.decoder = _make_decoder(list(self.encoder_dem.out_channels), decoder_dim)
            self.seg_head = _Head(decoder_dim, num_class)
        elif self.mode == 'ms':
            self.encoder_ms = enc(MS_CH)
            self.decoder = _make_decoder(list(self.encoder_ms.out_channels), decoder_dim)
            self.seg_head = _Head(decoder_dim, num_class)
        else:  # 'mid' : Simple / SRA / TTT
            self.encoder_dem = enc(DEM_CH)
            self.encoder_ms = enc(MS_CH)
            ch_dem = _stage_channels(self.encoder_dem, DEM_CH)
            self.fuse_modules = nn.ModuleList([
                FusionEncoder(
                    ch_dem[i],
                    head_dim=head_dim,
                    sr_ratio=sr_ratio,
                    channel_reduction_ratio=channel_reduction_ratio,
                    attention_type=self.attention_type,
                    height=img_size // (4 * 2 ** i), width=img_size // (4 * 2 ** i),
                    tokens_hidden_dim=tokens_hidden_dim,
                )
                for i in range(3)
            ])
            self.decoder = _make_decoder(list(self.encoder_dem.out_channels), decoder_dim)
            self.seg_head = _Head(decoder_dim, num_class)

    # ---- helpers -------------------------------------------------------------------------
    def encoder_param_ids(self):
        return {id(p) for n, m in self.named_children() if n.startswith('encoder') for p in m.parameters()}

    def set_probe(self, flag: bool):
        for m in self.modules():
            if isinstance(m, TTTCrossAttn):
                m.probe = flag
                m.last_probe = []

    def pop_probes(self):
        """List (per fusion scale) of the probe records collected since the last call."""
        if self.mode != 'mid' or self.attention_type != 'TTT':
            return []
        out = []
        for fm in self.fuse_modules:
            t = fm.cross_attn.ttt
            out.append(t.last_probe)
            t.last_probe = []
        return out

    # ---- forward -------------------------------------------------------------------------
    def _decode(self, decoder, head, raw_feats, size):
        seg = F.interpolate(decoder(raw_feats), size=size, mode='bilinear', align_corners=False)
        return head(seg)

    def forward(self, dtm: torch.Tensor, ms: torch.Tensor) -> Dict[str, object]:
        size = dtm.shape[-2:]
        if self.mode == 'early':
            raw = self.encoder_early(torch.cat([ms, dtm], dim=1))
            return {'seg_logits': self._decode(self.decoder, self.seg_head, raw, size)}
        if self.mode == 'dem':
            return {'seg_logits': self._decode(self.decoder, self.seg_head, self.encoder_dem(dtm), size)}
        if self.mode == 'ms':
            return {'seg_logits': self._decode(self.decoder, self.seg_head, self.encoder_ms(ms), size)}
        raw_dem, raw_ms = self.encoder_dem(dtm), self.encoder_ms(ms)
        if self.mode == 'late':
            l_dem = self._decode(self.decoder_dem, self.seg_head_dem, raw_dem, size)
            l_ms = self._decode(self.decoder_ms, self.seg_head_ms, raw_ms, size)
            return {'seg_logits': 0.5 * (l_dem + l_ms)}
        f_dem, f_ms = _stage_feats(raw_dem, DEM_CH), _stage_feats(raw_ms, MS_CH)
        fused = [self.fuse_modules[i](f_dem[i], f_ms[i]) for i in range(3)]
        return {'seg_logits': self._decode(self.decoder, self.seg_head, list(raw_dem[:2]) + fused, size)}


def build_model(variant: str, num_class: int, pretrained: bool = True, **kw) -> HedgeFormer:
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant '{variant}'. Choose from {list(VARIANTS)}")
    return HedgeFormer(num_class=num_class, encoder_weights='imagenet' if pretrained else None,
                       **VARIANTS[variant], **kw)


def load_checkpoint(model: nn.Module, path_or_state, map_location='cpu', verbose: bool = True):
    """STRICT load (the old code used strict=False, which silently left mismatched layers random).
    Only the lazily-created sinusoidal PE buffers of old SRA checkpoints are tolerated/ignored."""
    sd = torch.load(path_or_state, map_location=map_location) if isinstance(path_or_state, str) else path_or_state
    sd = {k: v for k, v in sd.items() if '.pe_' not in k}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'checkpoint does not match architecture: missing={missing[:5]}'
                           f'{"..." if len(missing) > 5 else ""} unexpected={unexpected[:5]}'
                           f'{"..." if len(unexpected) > 5 else ""}  (wrong variant?)')
    if verbose and isinstance(path_or_state, str):
        print(f'  loaded {path_or_state} (strict)')
    return model