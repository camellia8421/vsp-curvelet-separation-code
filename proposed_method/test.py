import os
import sys
import torch
import numpy as np
import logging
import pickle
import warnings
import traceback
import pandas as pd
from typing import Any, List, Tuple, Dict, Optional
from types import SimpleNamespace
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity as ssim

# ======================
# 核心导入：适配动态子带配置
# ======================
import utils
# 确保 utils 中有 compute_global_norm_params（兼容存根）
if not hasattr(utils, 'compute_global_norm_params'):
    utils.compute_global_norm_params = utils.compute_subband_norm_params

from utils import (
    init_subband_config,
    DivCurlSeparatedSubbandNet,
    denormalize_subband,
    compute_subband_norm_params,
    compute_global_norm_params
)
from cusdataloader import (
    SeismicCurveletDataloader, CurveletAwareTransform
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class TestOptions(SimpleNamespace):
    def __init__(self):
        # ======================
        # 1. 设备配置
        # ======================
        self.gpuid = 0
        self.cuda_visible_devices = "2"
        self.device = None

        # ======================
        # 2. 子带配置（与训练完全一致）
        # ======================
        self.subband_sizes = [
            (21, 171), (43, 151), (43, 151), (19, 343), (19, 343),
            (43, 151), (43, 151), (19, 343), (19, 343), (43, 279),
            (43, 279), (43, 279), (43, 279), (35, 343), (35, 343),
            (35, 343), (35, 343), (43, 279), (43, 279), (43, 279),
            (43, 279), (35, 343), (35, 343), (35, 343), (35, 343)
        ]
        self.scale_dir_counts = [1, 8, 16]

        # ======================
        # 3. 数据配置
        # ======================
        self.data_root = "./data"
        self.train_data_dir = "train_model_432_raw_subbands_512x64"
        self.test_data_dir = "test_model_432_raw_subbands_512x64"
        self.max_test_data = 140
        self.max_train_for_norm = 240
        self.data_start_test = 0
        self.data_start_train = 0
        self.jsample = 1

        # ======================
        # 4. 归一化配置（与训练一致）
        # ======================
        self.use_global_normalization = False
        self.high_freq_enhance_factor = 1.0
        self.curvelet_normalization = "mean_std"

        # ---------- 模型与逆变换路径 ----------
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_root = os.path.join(
            self.current_dir,
            "subband_separation_results_div_curl",
            "div_curl_model_train_model_62_raw_subbands_512x64_xxxxx_xxxxxx"
        )
        self.checkpoint_path = os.path.join(
            self.model_root,
            "best_epxxx_valxxx.pth"
        )
        self.inv_transform_info_path = os.path.join(self.model_root, "inverse_info.pkl")
        self.norm_params_path = None


        # ======================
        # 6. 预测配置
        # ======================
        self.batch_size = 2
        self.num_workers = 0
        self.pin_memory = False

        # ======================
        # 7. 输出配置
        # ======================
        self.train_model_id = 432
        self.test_model_id = 432
        self.epoch = 611
        self.output_dir = None
        self.save_subbands = False
        self.save_spatial = True
        self.save_format = "npy"
        self.spatial_image_size = (512, 64)
        self.compute_metrics = True

        # ======================
        # 8. Div/Curl损失配置（与训练一致）
        # ======================
        self.scale_weights = {0: 1.0, 1: 1.5, 2: 2.0}
        self.div_weight = 0.6
        self.curl_weight = 0.4

# ---------- 指标计算函数（增加SSIM） ----------
def calculate_metrics(target, pred):
    target_flat = target.flatten()
    pred_flat = pred.flatten()
    r2 = r2_score(target_flat, pred_flat)
    mse = np.mean((target_flat - pred_flat) ** 2)
    max_val = np.max(target_flat)
    psnr = 20 * np.log10(max_val / (np.sqrt(mse) + 1e-8)) if max_val > 0 else 0
    signal_power = np.mean(target_flat ** 2)
    snr = 10 * np.log10(signal_power / (mse + 1e-8))

    # SSIM 计算
    data_range = target.max() - target.min()
    if data_range == 0:
        ssim_val = 1.0
    else:
        # 防止 target 或 pred 为常数导致 SSIM 报错
        try:
            ssim_val = ssim(target, pred, data_range=data_range)
        except ValueError:
            # 如果图像尺寸太小，可增大窗口或使用全图计算
            ssim_val = ssim(target, pred, data_range=data_range, win_size=3)
    return {'r2': r2, 'psnr': psnr, 'snr': snr, 'ssim': ssim_val}

# ---------- 模型加载 ----------
def load_model(opt: TestOptions):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_visible_devices
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,garbage_collection_threshold:0.6'
    torch.cuda.set_device(opt.gpuid)
    opt.device = torch.device(f'cuda:{opt.gpuid}' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties(opt.device)
        gpu_mem_gb = gpu_prop.total_memory / 1024**3
        logger.info(f"✅ GPU信息：{gpu_prop.name} | 显存：{gpu_mem_gb:.1f}GB")
        torch.cuda.set_per_process_memory_fraction(0.8, device=opt.gpuid)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        logger.warning("⚠️ 未检测到GPU，使用CPU推理（速度较慢）")

    missing_files = []
    for path, desc in [(opt.checkpoint_path, "模型权重文件"), (opt.inv_transform_info_path, "逆变换信息文件")]:
        if not os.path.exists(path):
            missing_files.append(f"{desc}：{path}")
    if missing_files:
        raise FileNotFoundError(f"❌ 以下关键文件不存在：\n" + "\n".join(missing_files))

    logger.info(f"💾 加载Div/Curl分离模型：{os.path.basename(opt.checkpoint_path)}")
    checkpoint = torch.load(opt.checkpoint_path, map_location=opt.device)

    model = DivCurlSeparatedSubbandNet(
        in_channels=2,
        base_channels=64,
        subband_sizes=opt.subband_sizes,
        scale_dir_counts=opt.scale_dir_counts,
        dropout=0.0
    ).to(opt.device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        logger.info(f"📜 模型训练信息：")
        for key in ["epoch", "best_val_loss", "lr"]:
            if key in checkpoint:
                logger.info(f"  - {key}：{checkpoint[key]}")
    else:
        model.load_state_dict(checkpoint, strict=True)
        logger.warning("⚠️ Checkpoint中无model_state_dict，直接加载整个模型")

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ Div/Curl模型初始化完成：总参数量：{total_params:,d}（≈{total_params/1e6:.1f}M）")

    with open(opt.inv_transform_info_path, 'rb') as f:
        inv_transform_info = pickle.load(f)
    inv_transform_info.setdefault('scale_weights', opt.scale_weights)
    inv_transform_info.setdefault('div_weight', opt.div_weight)
    inv_transform_info.setdefault('curl_weight', opt.curl_weight)
    inv_transform_info.setdefault('high_freq_enhance_factor', opt.high_freq_enhance_factor)
    inv_transform_info.setdefault('use_global_normalization', opt.use_global_normalization)
    inv_transform_info.setdefault('norm_params', {})
    inv_transform_info.setdefault('subband_count', utils.SUBBAND_COUNT)

    logger.info(f"✅ 逆变换信息加载完成：归一化策略：{'全局' if inv_transform_info['use_global_normalization'] else '子带'}")
    return model, inv_transform_info

# ---------- 归一化参数补充 ----------
def supplement_norm_params(opt: TestOptions, inv_transform_info: Dict[str, Any]) -> Dict[str, Any]:
    required_components = ['curl', 'div']
    missing_components = [comp for comp in required_components if comp not in inv_transform_info['norm_params']]
    if not missing_components:
        logger.info("✅ curl/div归一化参数已存在，无需补充")
        return inv_transform_info

    logger.warning(f"⚠️ 缺失{missing_components}分量的归一化参数，从训练集重新计算...")
    train_dataset = SeismicCurveletDataloader(
        data_path=os.path.join(opt.data_root, opt.train_data_dir),
        max_data=opt.max_train_for_norm,
        data_start=opt.data_start_train,
        jsample=opt.jsample,
        transform=None,
        use_global_normalization=opt.use_global_normalization,
        high_freq_enhance_factor=opt.high_freq_enhance_factor,
        save_norm_params=True,
        norm_params_path=opt.norm_params_path,
        clip_outliers=False,
        fail_on_corrupt=True,
        subband_sizes=opt.subband_sizes,
        scale_dir_counts=opt.scale_dir_counts
    )
    inv_transform_info['norm_params'] = train_dataset.norm_params
    inv_transform_info['use_global_normalization'] = train_dataset.use_global_normalization
    inv_transform_info['high_freq_enhance_factor'] = train_dataset.high_freq_enhance_factor
    inv_transform_info['norm_strategy'] = train_dataset.curvelet_normalization if hasattr(train_dataset, "curvelet_normalization") else "mean_std"
    opt.norm_params_path = train_dataset.norm_params_path

    with open(opt.inv_transform_info_path, 'wb') as f:
        pickle.dump(inv_transform_info, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"✅ 已补充{missing_components}分量的归一化参数并保存至：{opt.inv_transform_info_path}")
    return inv_transform_info

def denormalize_subbands(subbands: List[np.ndarray], component: str, inv_transform_info: Dict[str, Any]) -> List[np.ndarray]:
    if component not in inv_transform_info.get('norm_params', {}):
        logger.warning(f"⚠️ 无{component}分量的归一化参数，返回原始数据")
        return subbands
    if len(subbands) != inv_transform_info.get('subband_count', utils.SUBBAND_COUNT):
        logger.warning(f"⚠️ 子带数量不匹配，返回原始数据")
        return subbands

    norm_params = inv_transform_info['norm_params']
    high_freq_enhance = inv_transform_info.get('high_freq_enhance_factor', 1.0)
    use_global = inv_transform_info.get('use_global_normalization', False)

    denorm_subbands = []
    for sb_idx, sb in enumerate(subbands):
        denorm_sb = denormalize_subband(
            sb_data=sb,
            norm_params=norm_params,
            sb_idx=sb_idx,
            component=component,
            high_freq_enhance=high_freq_enhance,
            use_global=use_global
        )
        denorm_subbands.append(denorm_sb.astype(np.float32))
    return denorm_subbands

def reconstruct_spatial_from_subbands(subbands: List[np.ndarray], image_size: Tuple[int, int]) -> Optional[np.ndarray]:
    try:
        import pyct
    except ImportError:
        logger.error("❌ 需安装pyct库以支持空间域重组")
        return None

    if len(subbands) != utils.SUBBAND_COUNT:
        logger.error(f"❌ 子带数{len(subbands)}≠预期{utils.SUBBAND_COUNT}")
        return None

    try:
        fdct2_obj = pyct.fdct2(
            n=image_size,
            nbs=3,
            nba=8,
            ac=True,
            vec=True,
            cpx=False
        )
        scale_dir_counts = [len(v) for v in utils.SUBBAND_SCALE_GROUP.values()]
        cl_struct = []
        ptr = 0
        for dir_count in scale_dir_counts:
            cl_struct.append(subbands[ptr:ptr+dir_count])
            ptr += dir_count
        clarray = fdct2_obj.vect(cl_struct)
        spatial_data = fdct2_obj.inv(clarray)
        spatial_data = spatial_data.reshape(image_size).astype(np.float32)
        return spatial_data
    except Exception as e:
        logger.error(f"❌ 空间域重组失败：{str(e)}", exc_info=False)
        return None

def save_pred_results(
    pred_s_wave: List[List[np.ndarray]],
    pred_p_wave: List[List[np.ndarray]],
    target_s_wave: List[List[np.ndarray]],
    target_p_wave: List[List[np.ndarray]],
    data_indices: List[int],
    opt: TestOptions,
    inv_transform_info: Dict[str, Any]
) -> None:
    s_wave_spatial_dir = os.path.join(opt.output_dir, "S_wave_spatial")
    p_wave_spatial_dir = os.path.join(opt.output_dir, "P_wave_spatial")
    os.makedirs(s_wave_spatial_dir, exist_ok=True)
    os.makedirs(p_wave_spatial_dir, exist_ok=True)

    metrics_list = []
    total_samples = len(data_indices)

    logger.info(f"📤 保存Div/Curl分离空间域结果至：{opt.output_dir}")
    for idx in tqdm(range(total_samples), desc="Processing samples", unit="sample"):
        s_subbands_pred = pred_s_wave[idx]
        p_subbands_pred = pred_p_wave[idx]
        s_subbands_target = target_s_wave[idx]
        p_subbands_target = target_p_wave[idx]
        data_idx = data_indices[idx]

        s_subbands_denorm_pred = denormalize_subbands(s_subbands_pred, 'curl', inv_transform_info)
        p_subbands_denorm_pred = denormalize_subbands(p_subbands_pred, 'div', inv_transform_info)
        s_subbands_denorm_target = denormalize_subbands(s_subbands_target, 'curl', inv_transform_info)
        p_subbands_denorm_target = denormalize_subbands(p_subbands_target, 'div', inv_transform_info)

        s_spatial_pred = reconstruct_spatial_from_subbands(s_subbands_denorm_pred, opt.spatial_image_size)
        s_spatial_target = reconstruct_spatial_from_subbands(s_subbands_denorm_target, opt.spatial_image_size)
        p_spatial_pred = reconstruct_spatial_from_subbands(p_subbands_denorm_pred, opt.spatial_image_size)
        p_spatial_target = reconstruct_spatial_from_subbands(p_subbands_denorm_target, opt.spatial_image_size)

        if any(x is None for x in [s_spatial_pred, s_spatial_target, p_spatial_pred, p_spatial_target]):
            logger.warning(f"样本{data_idx}空间域重建失败，跳过")
            continue

        if opt.save_spatial:
            np.save(os.path.join(s_wave_spatial_dir, f"S_wave_spatial_{data_idx}.npy"), s_spatial_pred)
            np.save(os.path.join(p_wave_spatial_dir, f"P_wave_spatial_{data_idx}.npy"), p_spatial_pred)
            np.save(os.path.join(s_wave_spatial_dir, f"S_wave_spatial_target_{data_idx}.npy"), s_spatial_target)
            np.save(os.path.join(p_wave_spatial_dir, f"P_wave_spatial_target_{data_idx}.npy"), p_spatial_target)

        if opt.compute_metrics:
            # 计算S波指标
            metrics_s = calculate_metrics(s_spatial_target, s_spatial_pred)
            # 计算P波指标
            metrics_p = calculate_metrics(p_spatial_target, p_spatial_pred)
            metrics_list.append({
                'filename': f"sample_{data_idx}",
                'r2p': metrics_p['r2'],
                'r2s': metrics_s['r2'],
                'psnr_p': metrics_p['psnr'],
                'psnr_s': metrics_s['psnr'],
                'snr_p': metrics_p['snr'],
                'snr_s': metrics_s['snr'],
                'ssim_p': metrics_p['ssim'],
                'ssim_s': metrics_s['ssim']
            })

    if opt.compute_metrics and metrics_list:
        df = pd.DataFrame(metrics_list)
        csv_path = os.path.join(opt.output_dir, 'metrics_all.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"✅ 指标结果已保存至：{csv_path}")

        print("\n------ 整体统计（所有样本） ------")
        print(f"样本总数: {len(df)}")
        for col in ['r2p', 'r2s', 'psnr_p', 'psnr_s', 'snr_p', 'snr_s', 'ssim_p', 'ssim_s']:
            mean_val = df[col].mean()
            median_val = df[col].median()
            max_val = df[col].max()
            min_val = df[col].min()
            print(f"{col}: 平均={mean_val:.6f}, 中位数={median_val:.6f}, 最大={max_val:.6f}, 最小={min_val:.6f}")

    logger.info(f"✅ 所有Div/Curl分离结果保存完成：{opt.output_dir}")

@torch.no_grad()
def test_model(opt: TestOptions) -> None:
    model, inv_transform_info = load_model(opt)
    inv_transform_info = supplement_norm_params(opt, inv_transform_info)

    test_transform = CurveletAwareTransform(flip_prob=0.0)
    test_dataset = SeismicCurveletDataloader(
        data_path=os.path.join(opt.data_root, opt.test_data_dir),
        max_data=opt.max_test_data,
        data_start=opt.data_start_test,
        jsample=opt.jsample,
        transform=test_transform,
        use_global_normalization=inv_transform_info.get('use_global_normalization', False),
        high_freq_enhance_factor=inv_transform_info.get('high_freq_enhance_factor', 1.0),
        save_norm_params=False,
        norm_params_path=opt.norm_params_path,
        clip_outliers=False,
        fail_on_corrupt=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        drop_last=False
    )
    logger.info(f"✅ 测试集加载完成：有效样本数{len(test_dataset)}")

    logger.info("\n🚀 开始Div/Curl分离预测...")
    all_pred_s_wave = []
    all_pred_p_wave = []
    all_target_s_wave = []
    all_target_p_wave = []
    all_data_indices = []

    for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Predicting batches", unit="batch")):
        input_sbs, target_sbs, data_indices, _ = batch_data
        B = len(data_indices)

        input_sbs = [sb.to(opt.device, non_blocking=opt.pin_memory) for sb in input_sbs]
        pred_sbs = model(input_sbs)

        for sample_idx in range(B):
            sample_curl_pred = []
            sample_div_pred = []
            sample_curl_target = []
            sample_div_target = []

            for sb_idx in range(utils.SUBBAND_COUNT):
                pred_sb = pred_sbs[sb_idx][sample_idx]
                curl_pred = pred_sb[0].cpu().numpy().reshape(utils.SUBBAND_SIZES[sb_idx])
                div_pred = pred_sb[1].cpu().numpy().reshape(utils.SUBBAND_SIZES[sb_idx])
                sample_curl_pred.append(curl_pred)
                sample_div_pred.append(div_pred)

                target_sb = target_sbs[sb_idx][sample_idx]
                curl_target = target_sb[0].cpu().numpy().reshape(utils.SUBBAND_SIZES[sb_idx])
                div_target = target_sb[1].cpu().numpy().reshape(utils.SUBBAND_SIZES[sb_idx])
                sample_curl_target.append(curl_target)
                sample_div_target.append(div_target)

            all_pred_s_wave.append(sample_curl_pred)
            all_pred_p_wave.append(sample_div_pred)
            all_target_s_wave.append(sample_curl_target)
            all_target_p_wave.append(sample_div_target)
            all_data_indices.append(data_indices[sample_idx].item())

        del input_sbs, pred_sbs, batch_data, target_sbs
        torch.cuda.empty_cache()

    save_pred_results(
        all_pred_s_wave,
        all_pred_p_wave,
        all_target_s_wave,
        all_target_p_wave,
        all_data_indices,
        opt,
        inv_transform_info
    )

    del model, test_loader, test_dataset
    torch.cuda.empty_cache()
    logger.info("\n🧹 Div/Curl分离测试完成，所有资源已释放")

def main():
    opt = TestOptions()
    init_subband_config(opt.subband_sizes, opt.scale_dir_counts)

    current_dir = os.getcwd()
    output_dir_name = f"Results_trainOn_m{opt.train_model_id}_testOn_m{opt.test_model_id}_epoch_{opt.epoch}"
    opt.output_dir = os.path.join(current_dir, output_dir_name)
    os.makedirs(opt.output_dir, exist_ok=True)

    logger.info("="*80)
    logger.info(f"📊 {utils.SUBBAND_COUNT}个子带级Div/Curl分离测试脚本")
    logger.info(f"📌 模型路径：{opt.checkpoint_path}")
    logger.info(f"📌 测试数据路径：{os.path.join(opt.data_root, opt.test_data_dir)}")
    logger.info(f"📌 输出路径：{opt.output_dir}")
    logger.info(f"📌 归一化策略：{'子带' if not opt.use_global_normalization else '全局'}-{opt.curvelet_normalization}")
    logger.info(f"📌 高频增强系数：{opt.high_freq_enhance_factor}")
    logger.info(f"📌 Div/Curl权重：{opt.div_weight}/{opt.curl_weight} | 尺度权重：{opt.scale_weights}")
    logger.info(f"📌 测试样本数：{opt.max_test_data} | 批次大小：{opt.batch_size}")
    logger.info("="*80)

    required_paths = [
        os.path.join(opt.data_root, opt.test_data_dir),
        os.path.join(opt.data_root, opt.train_data_dir),
        opt.checkpoint_path,
        opt.inv_transform_info_path
    ]
    missing_paths = [p for p in required_paths if not os.path.exists(p)]
    if missing_paths:
        logger.error("❌ 以下路径不存在：")
        for p in missing_paths:
            logger.error(f"  - {p}")
        return

    try:
        test_model(opt)
    except Exception as e:
        logger.error(f"\n❌ 测试过程中发生错误：{str(e)}", exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(1)

if __name__ == "__main__":
    main()