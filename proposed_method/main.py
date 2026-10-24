import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pickle
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional, Union
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

import utils
# 确保 utils 中有 compute_global_norm_params（兼容存根）
if not hasattr(utils, 'compute_global_norm_params'):
    utils.compute_global_norm_params = utils.compute_subband_norm_params

from utils import (
    init_subband_config,
    DivCurlSeparatedSubbandNet,
    DivCurlSeparateLoss,
    denormalize_subband
)
# 注意：不再从 utils 中 import SUBBAND_*，而是通过 utils. 访问

from cusdataloader import (
    SeismicCurveletDataloader,
    CurveletAwareTransform
)

# ==============================
# 1. 日志和路径配置
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# ==============================
# 2. 训练参数配置
# ==============================
class TrainConfig:
    # 设备配置
    gpuid = 0
    cuda_visible_devices = "0"
    device = None

    # ========== 子带配置 ==========
    subband_sizes = [
        (21, 171), (43, 151), (43, 151), (19, 343), (19, 343),
        (43, 151), (43, 151), (19, 343), (19, 343), (43, 279),
        (43, 279), (43, 279), (43, 279), (35, 343), (35, 343),
        (35, 343), (35, 343), (43, 279), (43, 279), (43, 279),
        (43, 279), (35, 343), (35, 343), (35, 343), (35, 343)
    ]
    scale_dir_counts = [1, 8, 16]

    # 数据配置
    data_root = "/home/litianhao/data"
    train_data_dir = "train_model_432_raw_subbands_512x64"
    val_data_dir = "test_model_432_raw_subbands_512x64"
    max_train_data = 240
    max_val_data = 50
    data_start_train = 0
    data_start_val = 0
    jsample = 1

    # 归一化配置
    use_global_normalization = False
    high_freq_enhance_factor = 1.0

    # 模型配置
    in_channels = 2
    common_channels = 64

    # 训练配置
    batch_size = 16
    gradient_accumulation_steps = 1
    num_workers = 4
    pin_memory = True
    epochs = 10000
    lr = 1e-4
    weight_decay = 1e-5
    patience = 200
    save_best_only = True
    use_amp = True

    # 损失配置
    scale_weights = {0: 1.0, 1: 1.5, 2: 2.0}
    div_weight = 0.6
    curl_weight = 0.4

    # 日志控制
    print_scale_freq = 50
    log_r2_per_scale = True
    log_div_curl_loss = True

cfg = TrainConfig()

# ==============================
# 结果保存路径
# ==============================
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_ROOT = "subband_separation_results_div_curl"
os.makedirs(RESULT_ROOT, exist_ok=True)

DATASET_ID = cfg.train_data_dir
MODEL_SAVE_DIR = os.path.join(RESULT_ROOT, f"div_curl_model_{DATASET_ID}_{TIMESTAMP}")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

TB_LOG_DIR = os.path.join(RESULT_ROOT, f"tb_logs_{DATASET_ID}_{TIMESTAMP}")
os.makedirs(TB_LOG_DIR, exist_ok=True)

# ==============================
# 3. R²计算函数（使用 utils. 访问全局变量）
# ==============================
def calculate_r2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    residual = torch.sum((pred_flat - target_flat) ** 2)
    target_mean = torch.mean(target_flat)
    total = torch.sum((target_flat - target_mean) ** 2)
    return 1 - residual / (total + eps)

def calculate_scale_r2(pred_sbs: List[torch.Tensor], target_sbs: List[torch.Tensor],
                       eps: float = 1e-8) -> Dict[int, float]:
    scale_r2 = {scale: 0.0 for scale in utils.SUBBAND_SCALE_GROUP.keys()}
    scale_count = {scale: 0 for scale in utils.SUBBAND_SCALE_GROUP.keys()}
    for sb_idx, (pred, target) in enumerate(zip(pred_sbs, target_sbs)):
        scale = utils.SB_IDX_TO_SCALE[sb_idx]
        r2 = calculate_r2(pred, target, eps)
        scale_r2[scale] += r2
        scale_count[scale] += 1
    for scale in scale_r2:
        scale_r2[scale] /= scale_count[scale] if scale_count[scale] > 0 else 1.0
    return scale_r2

def calculate_div_curl_r2(pred_sbs: List[torch.Tensor], target_sbs: List[torch.Tensor],
                          eps: float = 1e-8) -> Tuple[float, float]:
    all_div_pred, all_div_target, all_curl_pred, all_curl_target = [], [], [], []
    for pred, target in zip(pred_sbs, target_sbs):
        all_curl_pred.append(pred[:, 0:1].flatten())
        all_curl_target.append(target[:, 0:1].flatten())
        all_div_pred.append(pred[:, 1:2].flatten())
        all_div_target.append(target[:, 1:2].flatten())
    curl_r2 = calculate_r2(torch.cat(all_curl_pred), torch.cat(all_curl_target), eps)
    div_r2 = calculate_r2(torch.cat(all_div_pred), torch.cat(all_div_target), eps)
    return div_r2, curl_r2

def calculate_scale_div_curl_r2(pred_sbs: List[torch.Tensor], target_sbs: List[torch.Tensor]) -> Tuple[Dict[int, float], Dict[int, float]]:
    scale_div_r2 = {}
    scale_curl_r2 = {}
    for scale, sb_indices in utils.SUBBAND_SCALE_GROUP.items():
        scale_pred = [pred_sbs[i] for i in sb_indices]
        scale_target = [target_sbs[i] for i in sb_indices]
        div_r2, curl_r2 = calculate_div_curl_r2(scale_pred, scale_target)
        scale_div_r2[scale] = div_r2
        scale_curl_r2[scale] = curl_r2
    return scale_div_r2, scale_curl_r2

# ==============================
# 4. 增强版损失函数（使用 utils. 访问全局变量）
# ==============================
class OptimizedDivCurlSeparateLoss(DivCurlSeparateLoss):
    def forward(self, pred_sbs: list, target_sbs: list) -> Tuple[torch.Tensor, Dict[str, Any]]:
        total_loss = 0.0
        scale_loss_info = {
            "weighted_total": {},
            "div_raw_mse": {},
            "curl_raw_mse": {},
            "div_log_loss": {},
            "curl_log_loss": {}
        }
        for scale, sb_indices in utils.SUBBAND_SCALE_GROUP.items():
            scale_weight = self.scale_weights.get(scale, 1.0)
            scale_div_log, scale_curl_log = 0.0, 0.0
            scale_div_raw, scale_curl_raw = 0.0, 0.0
            for sb_idx in sb_indices:
                pred_curl = pred_sbs[sb_idx][:, 0:1]
                pred_div = pred_sbs[sb_idx][:, 1:2]
                target_curl = target_sbs[sb_idx][:, 0:1]
                target_div = target_sbs[sb_idx][:, 1:2]
                curl_raw = self.mse(pred_curl, target_curl)
                div_raw = self.mse(pred_div, target_div)
                scale_div_raw += div_raw.item()
                scale_curl_raw += curl_raw.item()
                scale_div_log += torch.log1p(div_raw)
                scale_curl_log += torch.log1p(curl_raw)
            avg_div_raw = scale_div_raw / len(sb_indices)
            avg_curl_raw = scale_curl_raw / len(sb_indices)
            avg_div_log = scale_div_log / len(sb_indices)
            avg_curl_log = scale_curl_log / len(sb_indices)
            weighted_loss = (avg_div_log * self.div_weight + avg_curl_log * self.curl_weight) * scale_weight
            total_loss += weighted_loss
            scale_loss_info["weighted_total"][scale] = weighted_loss.item()
            scale_loss_info["div_raw_mse"][scale] = avg_div_raw
            scale_loss_info["curl_raw_mse"][scale] = avg_curl_raw
            scale_loss_info["div_log_loss"][scale] = avg_div_log.item()
            scale_loss_info["curl_log_loss"][scale] = avg_curl_log.item()
        # 确保 total_loss 是 Tensor
        if not isinstance(total_loss, torch.Tensor):
            total_loss = torch.tensor(total_loss, device=next(self.parameters()).device)
        return total_loss, scale_loss_info

# ==============================
# 5. 数据加载（初始化全局变量）
# ==============================
def get_dataloaders() -> Tuple[DataLoader, DataLoader, SeismicCurveletDataloader]:
    # 初始化全局子带配置
    init_subband_config(cfg.subband_sizes, cfg.scale_dir_counts)
    logger.info(f"📐 全局子带配置初始化完成：{utils.SUBBAND_COUNT}个子带")

    train_transform = CurveletAwareTransform(flip_prob=0)
    train_dataset = SeismicCurveletDataloader(
        data_path=os.path.join(cfg.data_root, cfg.train_data_dir),
        max_data=cfg.max_train_data,
        data_start=cfg.data_start_train,
        jsample=cfg.jsample,
        transform=train_transform,
        use_global_normalization=cfg.use_global_normalization,
        high_freq_enhance_factor=cfg.high_freq_enhance_factor,
        save_norm_params=True,
        clip_outliers=False,
        fail_on_corrupt=True,
        subband_sizes=cfg.subband_sizes,
        scale_dir_counts=cfg.scale_dir_counts
    )

    val_dataset = SeismicCurveletDataloader(
        data_path=os.path.join(cfg.data_root, cfg.val_data_dir),
        max_data=cfg.max_val_data,
        data_start=cfg.data_start_val,
        jsample=cfg.jsample,
        transform=None,
        use_global_normalization=cfg.use_global_normalization,
        high_freq_enhance_factor=cfg.high_freq_enhance_factor,
        save_norm_params=False,
        norm_params_path=train_dataset.norm_params_path,
        clip_outliers=False,
        fail_on_corrupt=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False
    )

    logger.info(f"✅ 数据加载完成：训练集{len(train_dataset)}样本，验证集{len(val_dataset)}样本")
    return train_loader, val_loader, train_dataset

# ==============================
# 6. 模型和优化器初始化
# ==============================
def init_model_optimizer() -> Tuple[DivCurlSeparatedSubbandNet, optim.Optimizer, optim.lr_scheduler.ReduceLROnPlateau]:
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,garbage_collection_threshold:0.9'
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.6, device=cfg.gpuid)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    torch.cuda.set_device(cfg.gpuid)
    cfg.device = torch.device(f"cuda:{cfg.gpuid}" if torch.cuda.is_available() else "cpu")
    logger.info(f"✅ 使用设备：{cfg.device}")

    model = DivCurlSeparatedSubbandNet(
        in_channels=cfg.in_channels,
        base_channels=cfg.common_channels,
        subband_sizes=cfg.subband_sizes,
        scale_dir_counts=cfg.scale_dir_counts,
        dropout=0.05
    ).to(cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ 模型初始化完成：总参数量={total_params:,d}（≈{total_params/1e6:.1f}M）")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, verbose=True)
    return model, optimizer, scheduler

# ==============================
# 7. 训练函数
# ==============================
def train_one_epoch(
    model: DivCurlSeparatedSubbandNet,
    loader: DataLoader,
    criterion: OptimizedDivCurlSeparateLoss,
    optimizer: optim.Optimizer,
    epoch: int,
    writer: SummaryWriter,
    scaler: GradScaler
) -> float:
    model.train()
    total_loss, total_r2, total_div_r2, total_curl_r2 = 0.0, 0.0, 0.0, 0.0
    scale_r2_accum = {scale: 0.0 for scale in utils.SUBBAND_SCALE_GROUP.keys()}
    scale_div_r2_accum = {scale: 0.0 for scale in utils.SUBBAND_SCALE_GROUP.keys()}
    scale_curl_r2_accum = {scale: 0.0 for scale in utils.SUBBAND_SCALE_GROUP.keys()}
    batch_count = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=False)
    optimizer.zero_grad()

    for batch_idx, batch_data in enumerate(pbar):
        input_sbs, target_sbs, _, _ = batch_data
        input_sbs_device = [sb.to(cfg.device, non_blocking=cfg.pin_memory) for sb in input_sbs]
        target_sbs_device = [sb.to(cfg.device, non_blocking=cfg.pin_memory) for sb in target_sbs]

        with autocast(enabled=cfg.use_amp):
            pred_sbs = model(input_sbs_device)
            loss, scale_loss_info = criterion(pred_sbs, target_sbs_device)

        # 确保 loss 是 Tensor（防止空尺度组导致 float）
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, device=cfg.device)

        loss = loss / cfg.gradient_accumulation_steps
        scaler.scale(loss).backward()
        if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        batch_loss = loss.item() * cfg.gradient_accumulation_steps
        total_loss += batch_loss * input_sbs_device[0].shape[0]
        batch_count += 1

        # R²计算
        all_pred = torch.cat([p.flatten() for p in pred_sbs])
        all_target = torch.cat([t.flatten() for t in target_sbs_device])
        batch_r2 = calculate_r2(all_pred.unsqueeze(0), all_target.unsqueeze(0))
        batch_div_r2, batch_curl_r2 = calculate_div_curl_r2(pred_sbs, target_sbs_device)
        total_r2 += batch_r2
        total_div_r2 += batch_div_r2
        total_curl_r2 += batch_curl_r2

        if cfg.log_r2_per_scale:
            batch_scale_r2 = calculate_scale_r2(pred_sbs, target_sbs_device)
            scale_div_r2, scale_curl_r2 = calculate_scale_div_curl_r2(pred_sbs, target_sbs_device)
            for scale in utils.SUBBAND_SCALE_GROUP.keys():
                scale_r2_accum[scale] += batch_scale_r2[scale]
                scale_div_r2_accum[scale] += scale_div_r2[scale]
                scale_curl_r2_accum[scale] += scale_curl_r2[scale]

        if batch_idx % cfg.print_scale_freq == 0 and batch_idx > 0:
            logger.info(
                f"[训练] Batch {batch_idx} | 总损失：{batch_loss:.4f} | "
                f"尺度加权损失：{scale_loss_info['weighted_total']} | "
                f"Div R²：{batch_div_r2:.4f} | Curl R²：{batch_curl_r2:.4f}"
            )

        pbar.set_postfix({
            "loss": f"{batch_loss:.4f}",
            "r2": f"{batch_r2:.4f}",
            "div_r2": f"{batch_div_r2:.4f}",
            "curl_r2": f"{batch_curl_r2:.4f}"
        })

        del input_sbs_device, target_sbs_device, pred_sbs
        # torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader.dataset)
    avg_r2 = total_r2 / batch_count
    avg_div_r2 = total_div_r2 / batch_count
    avg_curl_r2 = total_curl_r2 / batch_count

    # TensorBoard
    global_step = epoch + 1
    writer.add_scalar("Train/TotalLoss", avg_loss, global_step)
    writer.add_scalar("Train/TotalR2", avg_r2, global_step)
    writer.add_scalar("Train/DivR2", avg_div_r2, global_step)
    writer.add_scalar("Train/CurlR2", avg_curl_r2, global_step)
    writer.add_scalar("Train/LearningRate", optimizer.param_groups[0]['lr'], global_step)

    for scale in scale_loss_info["weighted_total"]:
        writer.add_scalar(f"Train/Scale{scale}/WeightedLoss", scale_loss_info["weighted_total"][scale], global_step)
        writer.add_scalar(f"Train/Scale{scale}/DivRawMSE", scale_loss_info["div_raw_mse"][scale], global_step)
        writer.add_scalar(f"Train/Scale{scale}/CurlRawMSE", scale_loss_info["curl_raw_mse"][scale], global_step)
        writer.add_scalar(f"Train/Scale{scale}/DivLogLoss", scale_loss_info["div_log_loss"][scale], global_step)
        writer.add_scalar(f"Train/Scale{scale}/CurlLogLoss", scale_loss_info["curl_log_loss"][scale], global_step)

    if cfg.log_r2_per_scale:
        avg_scale_r2 = {s: v / batch_count for s, v in scale_r2_accum.items()}
        avg_scale_div_r2 = {s: v / batch_count for s, v in scale_div_r2_accum.items()}
        avg_scale_curl_r2 = {s: v / batch_count for s, v in scale_curl_r2_accum.items()}
        for scale in avg_scale_r2:
            writer.add_scalar(f"Train/Scale{scale}/TotalR2", avg_scale_r2[scale], global_step)
            writer.add_scalar(f"Train/Scale{scale}/DivR2", avg_scale_div_r2[scale], global_step)
            writer.add_scalar(f"Train/Scale{scale}/CurlR2", avg_scale_curl_r2[scale], global_step)

    logger.info(f"Epoch {epoch+1} | 训练损失：{avg_loss:.4f} | 训练R²：{avg_r2:.4f} | Div R²：{avg_div_r2:.4f} | Curl R²：{avg_curl_r2:.4f}")
    return avg_loss

# ==============================
# 8. 验证函数
# ==============================
@torch.no_grad()
def validate(
    model: DivCurlSeparatedSubbandNet,
    loader: DataLoader,
    criterion: OptimizedDivCurlSeparateLoss,
    epoch: int,
    writer: SummaryWriter,
    optimizer: optim.Optimizer
) -> float:
    model.eval()
    total_loss, total_r2, total_div_r2, total_curl_r2 = 0.0, 0.0, 0.0, 0.0
    scale_r2_accum = {scale: 0.0 for scale in utils.SUBBAND_SCALE_GROUP.keys()}
    scale_div_r2_accum = {scale: 0.0 for scale in utils.SUBBAND_SCALE_GROUP.keys()}
    scale_curl_r2_accum = {scale: 0.0 for scale in utils.SUBBAND_SCALE_GROUP.keys()}
    batch_count = 0

    pbar = tqdm(loader, desc="Validation", leave=False)
    for batch_idx, batch_data in enumerate(pbar):
        input_sbs, target_sbs, _, _ = batch_data
        input_sbs_device = [sb.to(cfg.device, non_blocking=cfg.pin_memory) for sb in input_sbs]
        target_sbs_device = [sb.to(cfg.device, non_blocking=cfg.pin_memory) for sb in target_sbs]

        pred_sbs = model(input_sbs_device)
        loss, scale_loss_info = criterion(pred_sbs, target_sbs_device)

        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, device=cfg.device)

        batch_loss = loss.item()
        total_loss += batch_loss * input_sbs_device[0].shape[0]
        batch_count += 1

        all_pred = torch.cat([p.flatten() for p in pred_sbs])
        all_target = torch.cat([t.flatten() for t in target_sbs_device])
        batch_r2 = calculate_r2(all_pred.unsqueeze(0), all_target.unsqueeze(0))
        batch_div_r2, batch_curl_r2 = calculate_div_curl_r2(pred_sbs, target_sbs_device)
        total_r2 += batch_r2
        total_div_r2 += batch_div_r2
        total_curl_r2 += batch_curl_r2

        if cfg.log_r2_per_scale:
            batch_scale_r2 = calculate_scale_r2(pred_sbs, target_sbs_device)
            scale_div_r2, scale_curl_r2 = calculate_scale_div_curl_r2(pred_sbs, target_sbs_device)
            for scale in utils.SUBBAND_SCALE_GROUP.keys():
                scale_r2_accum[scale] += batch_scale_r2[scale]
                scale_div_r2_accum[scale] += scale_div_r2[scale]
                scale_curl_r2_accum[scale] += scale_curl_r2[scale]

        pbar.set_postfix({
            "val_loss": f"{batch_loss:.4f}",
            "val_r2": f"{batch_r2:.4f}",
            "val_div_r2": f"{batch_div_r2:.4f}",
            "val_curl_r2": f"{batch_curl_r2:.4f}"
        })

        del input_sbs_device, target_sbs_device, pred_sbs
        # torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader.dataset)
    avg_r2 = total_r2 / batch_count
    avg_div_r2 = total_div_r2 / batch_count
    avg_curl_r2 = total_curl_r2 / batch_count

    global_step = epoch + 1
    writer.add_scalar("Validation/TotalLoss", avg_loss, global_step)
    writer.add_scalar("Validation/TotalR2", avg_r2, global_step)
    writer.add_scalar("Validation/DivR2", avg_div_r2, global_step)
    writer.add_scalar("Validation/CurlR2", avg_curl_r2, global_step)
    writer.add_scalar("Validation/LearningRate", optimizer.param_groups[0]['lr'], global_step)

    for scale in scale_loss_info["weighted_total"]:
        writer.add_scalar(f"Validation/Scale{scale}/WeightedLoss", scale_loss_info["weighted_total"][scale], global_step)
        writer.add_scalar(f"Validation/Scale{scale}/DivRawMSE", scale_loss_info["div_raw_mse"][scale], global_step)
        writer.add_scalar(f"Validation/Scale{scale}/CurlRawMSE", scale_loss_info["curl_raw_mse"][scale], global_step)
        writer.add_scalar(f"Validation/Scale{scale}/DivLogLoss", scale_loss_info["div_log_loss"][scale], global_step)
        writer.add_scalar(f"Validation/Scale{scale}/CurlLogLoss", scale_loss_info["curl_log_loss"][scale], global_step)

    if cfg.log_r2_per_scale:
        avg_scale_r2 = {s: v / batch_count for s, v in scale_r2_accum.items()}
        avg_scale_div_r2 = {s: v / batch_count for s, v in scale_div_r2_accum.items()}
        avg_scale_curl_r2 = {s: v / batch_count for s, v in scale_curl_r2_accum.items()}
        for scale in avg_scale_r2:
            writer.add_scalar(f"Validation/Scale{scale}/TotalR2", avg_scale_r2[scale], global_step)
            writer.add_scalar(f"Validation/Scale{scale}/DivR2", avg_scale_div_r2[scale], global_step)
            writer.add_scalar(f"Validation/Scale{scale}/CurlR2", avg_scale_curl_r2[scale], global_step)

    logger.info(f"Epoch {epoch+1} | 验证损失：{avg_loss:.4f} | 验证R²：{avg_r2:.4f} | Div R²：{avg_div_r2:.4f} | Curl R²：{avg_curl_r2:.4f} | 学习率：{optimizer.param_groups[0]['lr']:.6f}")
    return avg_loss

# ==============================
# 9. 保存逆变换信息
# ==============================
def save_inverse_transform_info(train_dataset: SeismicCurveletDataloader):
    inv_transform_info = {
        "norm_params": train_dataset.norm_params,
        "use_global_normalization": cfg.use_global_normalization,
        "high_freq_enhance_factor": cfg.high_freq_enhance_factor,
        "subband_sizes": utils.SUBBAND_SIZES,
        "subband_count": utils.SUBBAND_COUNT,
        "scale_dir_counts": cfg.scale_dir_counts
    }
    inv_info_path = os.path.join(MODEL_SAVE_DIR, "inverse_info.pkl")
    with open(inv_info_path, "wb") as f:
        pickle.dump(inv_transform_info, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"✅ 逆变换信息已保存：{inv_info_path}")

# ==============================
# 10. 主训练函数
# ==============================
def main_train():
    writer = SummaryWriter(log_dir=TB_LOG_DIR)
    logger.info(f"✅ TensorBoard日志路径：{TB_LOG_DIR}")

    try:
        train_loader, val_loader, train_dataset = get_dataloaders()
        model, optimizer, scheduler = init_model_optimizer()

        criterion = OptimizedDivCurlSeparateLoss(
            scale_weights=cfg.scale_weights,
            div_weight=cfg.div_weight,
            curl_weight=cfg.curl_weight
        ).to(cfg.device)

        scaler = GradScaler(enabled=cfg.use_amp)

        save_inverse_transform_info(train_dataset)

        best_val_loss = float("inf")
        best_epoch = 0
        early_stop_counter = 0

        logger.info("="*80)
        logger.info(f"🚀 开始曲波域Div/Curl分离训练（共{cfg.epochs}个epoch）")
        logger.info(f"📌 子带配置：{utils.SUBBAND_COUNT}个子带，尺度分布{cfg.scale_dir_counts}")
        logger.info(f"📌 归一化策略：{'全局' if cfg.use_global_normalization else '子带'}")
        logger.info(f"📌 损失权重：尺度{cfg.scale_weights} | Div{cfg.div_weight} | Curl{cfg.curl_weight}")
        logger.info(f"📌 批次配置：批次大小{cfg.batch_size} | 梯度累积{cfg.gradient_accumulation_steps} | 等效批次{cfg.batch_size*cfg.gradient_accumulation_steps}")
        logger.info(f"📌 学习率：初始{cfg.lr} | 权重衰减{cfg.weight_decay} | 混合精度{cfg.use_amp}")
        logger.info("="*80)

        for epoch in range(cfg.epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer, scaler)
            val_loss = validate(model, val_loader, criterion, epoch, writer, optimizer)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                early_stop_counter = 0
                model_save_path = os.path.join(MODEL_SAVE_DIR, f"best_ep{epoch+1}_val{val_loss:.4f}.pth")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "train_config": cfg.__dict__,
                    "subband_config": {
                        "sizes": utils.SUBBAND_SIZES,
                        "count": utils.SUBBAND_COUNT,
                        "scale_groups": utils.SUBBAND_SCALE_GROUP,
                        "idx_to_scale": utils.SB_IDX_TO_SCALE
                    }
                }, model_save_path)
                logger.info(f"✅ 保存最优模型：{model_save_path}（验证损失：{val_loss:.4f}）")
                if cfg.save_best_only:
                    import glob
                    old_models = glob.glob(os.path.join(MODEL_SAVE_DIR, "best_ep*_val*.pth"))
                    for old_model in old_models:
                        if old_model != model_save_path:
                            os.remove(old_model)
            else:
                early_stop_counter += 1
                if early_stop_counter % 20 == 0 or early_stop_counter >= cfg.patience:
                    logger.info(f"⚠️ 早停计数器：{early_stop_counter}/{cfg.patience}")
                if early_stop_counter >= cfg.patience:
                    logger.info(f"🎉 早停触发！最优epoch：{best_epoch+1}，最优验证损失：{best_val_loss:.4f}")
                    break

        logger.info("="*80)
        logger.info(f"🎯 训练完成！")
        logger.info(f"📊 最优结果：Epoch {best_epoch+1} | 验证损失 {best_val_loss:.4f}")
        logger.info(f"📂 模型保存路径：{MODEL_SAVE_DIR}")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"❌ 训练过程中发生错误：{str(e)}", exc_info=True)
        raise
    finally:
        writer.close()
        logger.info(f"✅ TensorBoard writer已关闭")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main_train()