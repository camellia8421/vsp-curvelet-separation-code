"""
工具类文件：曲波域波场分离核心工具（v2 协同网络版）
新增：
- 三组共享编码器（低/中/高尺度）
- 跨尺度上下文融合（自上而下+自下而上）
- 尺度内方向图自注意力（可选，预留）
- 依旧兼容任意子带尺寸与尺度分布
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Tuple, Any

# ======================== 全局配置 ========================
SUBBAND_COUNT = 0
SUBBAND_SIZES = []
SUBBAND_SCALE_GROUP = {}
SB_IDX_TO_SCALE = {}

def init_subband_config(subband_sizes, scale_dir_counts):
    global SUBBAND_COUNT, SUBBAND_SIZES, SUBBAND_SCALE_GROUP, SB_IDX_TO_SCALE
    SUBBAND_SIZES = subband_sizes.copy()
    SUBBAND_COUNT = len(subband_sizes)
    SUBBAND_SCALE_GROUP = {}
    SB_IDX_TO_SCALE = {}
    ptr = 0
    for scale_idx, dir_count in enumerate(scale_dir_counts):
        sb_indices = list(range(ptr, ptr + dir_count))
        SUBBAND_SCALE_GROUP[scale_idx] = sb_indices
        for sb_idx in sb_indices:
            SB_IDX_TO_SCALE[sb_idx] = scale_idx
        ptr += dir_count
    assert ptr == SUBBAND_COUNT, "子带总数与尺度分布不匹配"
    print(f"✅ 子带配置初始化完成：共 {SUBBAND_COUNT} 个子带，尺度分布 {scale_dir_counts}")

def _validate_subband_config():
    if SUBBAND_COUNT == 0:
        print("⚠️ 子带配置未初始化")
        return
    assert len(SUBBAND_SIZES) == SUBBAND_COUNT
    assert set(SB_IDX_TO_SCALE.keys()) == set(range(SUBBAND_COUNT))

# ======================== 归一化工具 ========================
def compute_subband_norm_params(dataset, components, curvelet_normalization="mean_std"):
    norm_params = {}
    for comp_idx, comp in enumerate(components):
        norm_params[comp] = {}
        for sb_idx in range(SUBBAND_COUNT):
            sb_data = np.concatenate([sample[comp_idx][sb_idx].flatten() for sample in dataset])
            mean = np.mean(sb_data)
            std = np.std(sb_data) + 1e-8
            norm_params[comp][sb_idx] = (float(mean), float(std))
    return norm_params

def normalize_subband(sb_data, norm_params, sb_idx, component, high_freq_enhance=1.0, use_global=False):
    if SB_IDX_TO_SCALE[sb_idx] == 2:
        sb_data = sb_data * high_freq_enhance
    if use_global:
        mean, std = norm_params[component]
    else:
        mean, std = norm_params[component][sb_idx]
    return ((sb_data - mean) / std).astype(np.float32)

def denormalize_subband(sb_data, norm_params, sb_idx, component, high_freq_enhance=1.0, use_global=False):
    if use_global:
        mean, std = norm_params[component]
    else:
        mean, std = norm_params[component][sb_idx]
    denormalized = (sb_data * std + mean)
    if SB_IDX_TO_SCALE[sb_idx] == 2:
        denormalized = denormalized / high_freq_enhance
    return denormalized.astype(np.float32)

def augment_high_freq_subbands(sb_list, aug_strength=0.01):
    augmented = []
    for sb_idx, sb_data in enumerate(sb_list):
        scale = SB_IDX_TO_SCALE[sb_idx]
        if scale == 2:
            noise = torch.randn_like(sb_data) * aug_strength
            scale_factor = 1.0 + (torch.rand(1, device=sb_data.device) * 0.1 - 0.05)
            scale_factor = scale_factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            sb_data = sb_data * scale_factor + noise
        augmented.append(sb_data)
    return augmented

# ======================== SSIM 损失 ========================
def ssim(img1, img2, window_size=7, size_average=True):
    def _gaussian_kernel(window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / (2*sigma**2)) for x in range(window_size)])
        return (gauss / gauss.sum()).unsqueeze(1)
    window = _gaussian_kernel(window_size, 1.5)
    window = window.mm(window.t()).unsqueeze(0).unsqueeze(0).to(img1.device)
    channels = img1.size(1)
    window = window.expand(channels, 1, window_size, window_size).contiguous()
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channels) - mu1_mu2
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

class SpatialSSIMLoss(nn.Module):
    def __init__(self, window_size=7):
        super().__init__()
        self.window_size = window_size
    def forward(self, pred_spatial, target_spatial):
        return 1 - ssim(pred_spatial, target_spatial, self.window_size)

# ======================== 基础组件 ========================
def _get_group_norm(num_channels, base_group_size=16):
    groups = max(1, num_channels // base_group_size)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)

class ConvBlock(nn.Module):
    """标准卷积块，用于共享编码器"""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _get_group_norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _get_group_norm(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class CrossScaleFusionUp(nn.Module):
    """自上而下：低分辨率特征上采样后与高分辨率特征融合"""
    def __init__(self, low_channels, high_channels, fused_channels):
        super().__init__()
        self.low_proj = nn.Conv2d(low_channels, fused_channels, 1)
        self.high_proj = nn.Conv2d(high_channels, fused_channels, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_channels*2, fused_channels, 3, padding=1, bias=False),
            _get_group_norm(fused_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, high_feat, low_feat):
        low_up = F.interpolate(low_feat, size=high_feat.shape[2:], mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([self.high_proj(high_feat), self.low_proj(low_up)], dim=1))

class CrossScaleFusionDown(nn.Module):
    """自下而上：高分辨率特征下采样后与低分辨率特征融合"""
    def __init__(self, high_channels, low_channels, fused_channels):
        super().__init__()
        self.high_proj = nn.Conv2d(high_channels, fused_channels, 3, stride=2, padding=1, bias=False)
        self.low_proj = nn.Conv2d(low_channels, fused_channels, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_channels*2, fused_channels, 3, padding=1, bias=False),
            _get_group_norm(fused_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, low_feat, high_feat):
        high_down = self.high_proj(high_feat)
        if high_down.shape[2:] != low_feat.shape[2:]:
            high_down = F.interpolate(high_down, size=low_feat.shape[2:], mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([self.low_proj(low_feat), high_down], dim=1))

class DirectionalGraphAttention(nn.Module):
    """尺度内方向交互"""
    def __init__(self, num_directions, feature_dim):
        super().__init__()
        self.num_dirs = num_directions
        self.feat_per_dir = feature_dim // num_directions
        self.scale = nn.Parameter(torch.ones(1, num_directions, 1, 1, 1) * 0.1)
        self.out_proj = nn.Conv2d(feature_dim, feature_dim, 1)

    def forward(self, x):
        # x: (B, C, H, W) where C = num_dirs * feat_per_dir
        B, C, H, W = x.shape
        x_rs = x.view(B, self.num_dirs, self.feat_per_dir, H, W)
        # 方向间注意力：对每个空间位置计算方向间的相关性
        # 简化：使用全局平均方向特征作为query
        dir_feat = x_rs.mean(dim=(-1,-2))  # (B, num_dirs, feat_per_dir)
        attn = torch.softmax(torch.einsum('bdf,bef->bde', dir_feat, dir_feat) * self.scale.squeeze(-1).squeeze(-1), dim=-1)
        # 用注意力聚合方向特征
        x_rs_attn = torch.einsum('bde,befhw->bdfhw', attn, x_rs)
        x_out = x_rs_attn.reshape(B, C, H, W)
        return self.out_proj(x_out) + x  # 残差连接

# ======================== 核心网络 ========================
class CollaborativeCurveletNet(nn.Module):
    def __init__(
        self,
        in_channels=2,
        base_channels=128,         
        subband_sizes: List[Tuple[int, int]] = None,
        scale_dir_counts: List[int] = None,
        dropout=0.05
    ):
        super().__init__()
        # 初始化子带配置（若未全局初始化则使用传入参数）
        if subband_sizes is not None:
            self.subband_sizes = subband_sizes
            self.subband_count = len(subband_sizes)
            if scale_dir_counts is not None:
                self.sb_idx_to_scale = {}
                ptr = 0
                for s_idx, dir_cnt in enumerate(scale_dir_counts):
                    for _ in range(dir_cnt):
                        self.sb_idx_to_scale[ptr] = s_idx
                        ptr += 1
            else:
                self.sb_idx_to_scale = {i: 0 for i in range(self.subband_count)}
        else:
            assert SUBBAND_COUNT > 0, "请先全局初始化子带配置，或传入 subband_sizes"
            self.subband_sizes = SUBBAND_SIZES
            self.subband_count = SUBBAND_COUNT
            self.sb_idx_to_scale = SB_IDX_TO_SCALE

        # 按尺寸分组
        size_to_indices = {}
        for idx, (h, w) in enumerate(self.subband_sizes):
            size_to_indices.setdefault((h, w), []).append(idx)
        self.size_groups = size_to_indices
        # 路径定义：两条主要分辨率链
        path1_sizes = [(21,171), (43,151), (43,279)]
        path2_sizes = [(21,171), (19,343), (35,343)]
        self.paths = [path1_sizes, path2_sizes]

        # 计算各编码器输入通道数（同一路径上同尺寸组的子带数一致）
        # 低频：(21,171) 仅尺度0的1个子带 → 2通道
        # 中频：(43,151) 包含尺度1的4个子带 → 8通道；(19,343) 同理也是4个→8通道
        # 高频：(43,279) 包含尺度2的8个子带 →16通道；(35,343) 8个→16通道
        self.enc_low = self._make_deep_encoder(2, base_channels, dropout)
        self.enc_mid = self._make_deep_encoder(8, base_channels, dropout)   # 尺度1共享
        self.enc_high = self._make_deep_encoder(16, base_channels, dropout) # 尺度2共享

        # 为每条路径构建独立的跨尺度融合模块列表
        self.path_fusions = nn.ModuleDict()
        for pid, path in enumerate(self.paths):
            up_mods = nn.ModuleList()
            down_mods = nn.ModuleList()
            for i in range(len(path)-1):
                up_mods.append(CrossScaleFusionUp(base_channels, base_channels, base_channels))
            for i in range(len(path)-1, 0, -1):
                down_mods.append(CrossScaleFusionDown(base_channels, base_channels, base_channels))
            self.path_fusions[f'path{pid}'] = nn.ModuleDict({'up': up_mods, 'down': down_mods})

        # 方向图注意力（每个尺度内，根据方向数动态调整）
        self.dir_attn = nn.ModuleDict()
        for size, indices in self.size_groups.items():
            scale = self.sb_idx_to_scale[indices[0]]
            if scale == 0: continue
            num_dirs = len(indices)
            self.dir_attn[str(size)] = DirectionalGraphAttention(num_dirs, base_channels)

        # 轻量输出头（每个子带独立，结合原始输入与组特征）
        self.output_heads = nn.ModuleList()
        for sb_idx in range(self.subband_count):
            self.output_heads.append(
                nn.Sequential(
                    nn.Conv2d(base_channels + in_channels, base_channels//2, 3, padding=1, bias=False),
                    _get_group_norm(base_channels//2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(base_channels//2, 2, 1)
                )
            )

    def _make_deep_encoder(self, in_ch, base_ch, dropout):
        """深度编码器：三层 ConvBlock，通道变化 base->2*base->base"""
        return nn.Sequential(
            ConvBlock(in_ch, base_ch, dropout),
            ConvBlock(base_ch, base_ch*2, dropout),
            ConvBlock(base_ch*2, base_ch, dropout),
        )

    def forward(self, input_sbs: List[torch.Tensor]) -> List[torch.Tensor]:
        # 1. 按尺寸组编码
        size_features = {}
        for size, indices in self.size_groups.items():
            group_tensor = torch.cat([input_sbs[i] for i in indices], dim=1)  # (B, 2*N, H, W)
            scale = self.sb_idx_to_scale[indices[0]]
            if scale == 0:
                feat = self.enc_low(group_tensor)
            elif scale == 1:
                feat = self.enc_mid(group_tensor)
            else:  # scale == 2
                feat = self.enc_high(group_tensor)
            size_features[size] = feat

        # 2. 方向交互（在每个组内进行）
        for size, feat in size_features.items():
            if str(size) in self.dir_attn:
                size_features[size] = self.dir_attn[str(size)](feat)

        # 3. 跨尺度上下文融合（沿两条路径）
        for pid, path in enumerate(self.paths):
            fusions = self.path_fusions[f'path{pid}']
            # 自上而下增强
            for i in range(len(path)-1):
                low_size = path[i]
                high_size = path[i+1]
                feat_low = size_features[low_size]
                feat_high = size_features[high_size]
                feat_high_new = fusions['up'][i](feat_high, feat_low)
                size_features[high_size] = feat_high_new
            # 自下而上增强
            for i in range(len(path)-1, 0, -1):
                low_size = path[i-1]
                high_size = path[i]
                feat_low = size_features[low_size]
                feat_high = size_features[high_size]   # 已增强后的高频
                feat_low_new = fusions['down'][len(path)-1 - i](feat_low, feat_high)
                size_features[low_size] = feat_low_new

        # 4. 生成各子带输出
        outputs = []
        for sb_idx in range(self.subband_count):
            sb_input = input_sbs[sb_idx]                       # (B,2,H,W)
            size = self.subband_sizes[sb_idx]
            feat = size_features[size]                         # (B, base_ch, H, W)
            combined = torch.cat([feat, sb_input], dim=1)
            out = self.output_heads[sb_idx](combined)          # (B,2,H,W) [curl, div]
            outputs.append(out)
        return outputs

DivCurlSeparatedSubbandNet = CollaborativeCurveletNet

# ======================== 损失函数 ========================
class DivCurlSeparateLoss(nn.Module):
    def __init__(self, scale_weights=None, div_weight=0.5, curl_weight=0.5):
        super().__init__()
        if scale_weights is None:
            if SUBBAND_SCALE_GROUP:
                self.scale_weights = {s: 1.0 + s * 0.5 for s in SUBBAND_SCALE_GROUP.keys()}
            else:
                self.scale_weights = {0: 1.0}
        else:
            self.scale_weights = scale_weights
        self.div_weight = div_weight
        self.curl_weight = curl_weight
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, pred_sbs, target_sbs):
        total_loss = 0.0
        scale_group = SUBBAND_SCALE_GROUP if SUBBAND_SCALE_GROUP else {0: list(range(len(pred_sbs)))}
        for scale, sb_indices in scale_group.items():
            sw = self.scale_weights.get(scale, 1.0)
            div_loss = 0.0
            curl_loss = 0.0
            for sb_idx in sb_indices:
                pred_curl = pred_sbs[sb_idx][:, 0:1]
                pred_div = pred_sbs[sb_idx][:, 1:2]
                target_curl = target_sbs[sb_idx][:, 0:1]
                target_div = target_sbs[sb_idx][:, 1:2]
                curl_loss += torch.log1p(self.mse(pred_curl, target_curl))
                div_loss += torch.log1p(self.mse(pred_div, target_div))
            avg_loss = (div_loss * self.div_weight + curl_loss * self.curl_weight) / len(sb_indices)
            total_loss += avg_loss * sw
        return total_loss

# ======================== 调试入口 ========================
if __name__ == "__main__":
    # 子带配置
    subband_sizes = [
        (21, 171), (43, 151), (43, 151), (19, 343), (19, 343),
        (43, 151), (43, 151), (19, 343), (19, 343), (43, 279),
        (43, 279), (43, 279), (43, 279), (35, 343), (35, 343),
        (35, 343), (35, 343), (43, 279), (43, 279), (43, 279),
        (43, 279), (35, 343), (35, 343), (35, 343), (35, 343)
    ]
    scale_dist = [1, 8, 16]
    init_subband_config(subband_sizes, scale_dist)

    # 模型实例化
    model = CollaborativeCurveletNet(in_channels=2, base_channels=64,
                                     subband_sizes=subband_sizes, scale_dir_counts=scale_dist)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量：{total_params:,} ≈ {total_params/1e6:.2f}M")

    # 前向传播测试
    dummy_sbs = [torch.randn(2, 2, h, w) for (h, w) in subband_sizes]
    with torch.no_grad():
        output_sbs = model(dummy_sbs)
    print(f"输入子带数: {len(dummy_sbs)}, 输出子带数: {len(output_sbs)}")
    for i in [0, 1, 9, 13, 24]:
        print(f"子带{i:2d} 输出形状: {output_sbs[i].shape}")