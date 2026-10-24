import os
import re
import numpy as np
import pyct
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================
# 配置参数
# ============================================================
TARGET_SHAPE = (512, 64)
TARGET_DT = 0.004
DEPTH_START = 1110.94
DEPTH_END = 2370.94

NOISE_LEVELS = [0.0, 0.5, 1.0] 

# 指向第二段代码中已经验证正确的预处理数据目录
PREPROCESSED_DATA_DIR = "./data/test_model_62_preprocessed_512x64"

OUTPUT_DIR = "./data/test_model_62_noisy_subbands_512x64"
SPATIAL_DIR = os.path.join(OUTPUT_DIR, "spatial")
os.makedirs(SPATIAL_DIR, exist_ok=True)

COMPONENTS = ["vx", "vz", "div", "curl"]
NOISY_COMPONENTS = ["vx", "vz"]

# 曲波变换初始化
nbs, nba = 3, 8
fdct2_obj = pyct.fdct2(n=TARGET_SHAPE, nbs=nbs, nba=nba,
                        ac=True, vec=True, cpx=False)

def get_all_sample_ids():
    """自动扫描预处理目录下的所有样本 ID"""
    search_dir = os.path.join(PREPROCESSED_DATA_DIR, "vx")
    if not os.path.exists(search_dir):
        search_dir = PREPROCESSED_DATA_DIR
    
    files = [f for f in os.listdir(search_dir) if f.startswith("vx_") and f.endswith(".npy")]
    sample_ids = []
    for f in files:
        nums = re.findall(r'\d+', f)
        if nums:
            sample_ids.append(int(nums[0]))
    return sorted(list(set(sample_ids)))


def load_preprocessed_data(comp, sid):
    file_path = os.path.join(PREPROCESSED_DATA_DIR, f"{comp}_{sid}.npy")
    if not os.path.exists(file_path):
        file_path = os.path.join(PREPROCESSED_DATA_DIR, comp, f"{comp}_{sid}.npy")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到预处理文件：{file_path}，请检查路径")

    data = np.load(file_path).astype(np.float32)
    if data.shape != TARGET_SHAPE:
        raise ValueError(f"文件 {file_path} 尺寸为 {data.shape}，不符合目标尺寸 {TARGET_SHAPE}")
    return data


def add_gaussian_noise(data, noise_level):
    if noise_level == 0.0:
        return data.copy()
    signal_std = np.std(data)
    noise_std = noise_level * signal_std
    noise = np.random.randn(*data.shape).astype(np.float32) * noise_std
    return data + noise


def save_spatial_and_curvelet(data, comp, noise_tag, sid):
    # 保存空间域 .npy
    spatial_folder = os.path.join(SPATIAL_DIR, comp)
    os.makedirs(spatial_folder, exist_ok=True)
    spatial_filename = f"{comp}_{sid}_{noise_tag}.npy"
    np.save(os.path.join(spatial_folder, spatial_filename), data)

    # 曲波变换并保存
    clarray = fdct2_obj.fwd(data)
    cl_struct = fdct2_obj.struct(clarray)
    raw_subbands = []
    for scale in cl_struct:
        for subband in scale:
            raw_subbands.append(subband)
    curvelet_data = np.array(raw_subbands, dtype=object)

    curvelet_folder = os.path.join(OUTPUT_DIR, comp, noise_tag)
    os.makedirs(curvelet_folder, exist_ok=True)
    curvelet_filename = f"{comp}_{sid}.npy"
    np.save(os.path.join(curvelet_folder, curvelet_filename), curvelet_data)


def plot_noisy_comparison(clean_vx, noisy_vx, clean_vz, noisy_vz, noise_level, sid, save_dir):
    vmax = max(np.max(np.abs(clean_vx)), np.max(np.abs(clean_vz)),
               np.max(np.abs(noisy_vx)), np.max(np.abs(noisy_vz)))
    T_END = TARGET_SHAPE[0] * TARGET_DT
    extent = [DEPTH_START, DEPTH_END, T_END, 0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    
    axes[0, 0].imshow(clean_vx, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax, extent=extent)
    axes[0, 0].set_title(f"Sample {sid} - Clean Vx")
    axes[0, 0].set_ylabel("Time (s)")

    axes[0, 1].imshow(noisy_vx, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax, extent=extent)
    axes[0, 1].set_title(f"Noisy Vx (noise {noise_level*100:.0f}%)")

    axes[1, 0].imshow(clean_vz, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax, extent=extent)
    axes[1, 0].set_title("Clean Vz")
    axes[1, 0].set_ylabel("Time (s)")
    axes[1, 0].set_xlabel("Depth (m)")

    im11 = axes[1, 1].imshow(noisy_vz, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax, extent=extent)
    axes[1, 1].set_title(f"Noisy Vz (noise {noise_level*100:.0f}%)")
    axes[1, 1].set_xlabel("Depth (m)")

    fig.colorbar(im11, ax=axes, shrink=0.8, label='Amplitude')
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"sample_{sid}_noisy_noise{noise_level}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    sample_ids = get_all_sample_ids()
    print(f"检测到测试集共有 {len(sample_ids)} 个样本，ID 列表：{sample_ids}")
    print(f"噪声水平配置：{NOISE_LEVELS}")

    plot_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # 遍历所有样本进行处理
    for sid in tqdm(sample_ids, desc="Processing Samples"):
        # 1. 加载当前样本的所有分量
        aligned_clean = {}
        for comp in COMPONENTS:
            aligned_clean[comp] = load_preprocessed_data(comp, sid)

        # 2. 保存干净版本（空间域 + 曲波域）
        for comp in COMPONENTS:
            save_spatial_and_curvelet(aligned_clean[comp], comp, "clean", sid)

        # 3. 对不同噪声水平进行处理
        for nl in NOISE_LEVELS:
            tag = f"noise{nl}"

            noisy_vx = add_gaussian_noise(aligned_clean["vx"], nl)
            noisy_vz = add_gaussian_noise(aligned_clean["vz"], nl)

            for comp in COMPONENTS:
                if comp in NOISY_COMPONENTS:
                    data = noisy_vx if comp == "vx" else noisy_vz
                else:
                    data = aligned_clean[comp].copy()
                save_spatial_and_curvelet(data, comp, tag, sid)

            # 可选：仅为特定样本（如 ID=12）或全部样本生成可视化对比图
            # 如果样本量很大，建议限制只为部分样本绘图以节省时间
            if sid == 12 or len(sample_ids) <= 20:
                plot_noisy_comparison(aligned_clean["vx"], noisy_vx,
                                      aligned_clean["vz"], noisy_vz,
                                      nl, sid, plot_dir)

    print("\n===== 全部测试集处理完成 =====")
    print(f"结果根目录：{OUTPUT_DIR}")
    print(f"空间域数据位于：{SPATIAL_DIR}")
    print(f"曲波域数据位于：{OUTPUT_DIR}/<component>/<noise_tag>/")