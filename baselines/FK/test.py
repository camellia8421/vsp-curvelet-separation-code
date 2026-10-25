import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib.colors import TwoSlopeNorm

# ====================== 配置区 ======================
SAMPLE_ID = 107
NOISE_LEVEL = 1.0

if NOISE_LEVEL == 0.0:
    NOISE_TAG_SPATIAL = "noise0.0"
elif NOISE_LEVEL == 0.5:
    NOISE_TAG_SPATIAL = "noise0.5"
elif NOISE_LEVEL == 1.0:
    NOISE_TAG_SPATIAL = "noise1.0"
else:
    NOISE_TAG_SPATIAL = f"noise{NOISE_LEVEL}"

SPATIAL_DIR = "./data/test_model_62_noisy_subbands_512x64/spatial"

NT, NR = 512, 64
DEPTH_START = 1110.94
DEPTH_END   = 2370.94
DT          = 0.004          
DX          = (DEPTH_END - DEPTH_START) / (NR - 1) 

TIME_AXIS   = np.linspace(0, NT * DT, NT)
DEPTH_AXIS  = np.linspace(DEPTH_START, DEPTH_END, NR)

SAVE_BASENAME = f"./Figure_FK_FanFilter_Noise{NOISE_LEVEL}"
# ===================================================

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "figure.titlesize": 8,
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

def fk_fan_filter_separation(vz, vx, dt, dx):
    nt, nr = vz.shape
    
    VZ = np.fft.fft2(vz)
    VX = np.fft.fft2(vx)
    
    f = np.fft.fftfreq(nt, d=dt)
    k = np.fft.fftfreq(nr, d=dx)
    
    F, K = np.meshgrid(f, k, indexing='ij')
    
    with np.errstate(divide='ignore', invalid='ignore'):
        v_app = np.abs(F / (K + 1e-10))
    
    v_split = 3000.0  
    transition = 800.0
    
    p_mask = 1.0 / (1.0 + np.exp(-(v_app - v_split) / transition))
    s_mask = 1.0 - p_mask
    
    VZ_p = VZ * p_mask
    VX_p = VX * p_mask
    VZ_s = VZ * s_mask
    VX_s = VX * s_mask
    
    p_est = np.real(np.fft.ifft2(VZ_p))
    s_est = np.real(np.fft.ifft2(VX_s))
    
    rms_p_target = np.sqrt(np.mean(vz**2))
    rms_s_target = np.sqrt(np.mean(vx**2))
    
    p_est = p_est * (rms_p_target / (np.sqrt(np.mean(p_est**2)) + 1e-8)) * 0.9
    s_est = s_est * (rms_s_target / (np.sqrt(np.mean(s_est**2)) + 1e-8)) * 0.9
    
    return p_est, s_est

if __name__ == "__main__":
    vz_path = os.path.join(SPATIAL_DIR, "vz", f"vz_{SAMPLE_ID}_{NOISE_TAG_SPATIAL}.npy")
    vx_path = os.path.join(SPATIAL_DIR, "vx", f"vx_{SAMPLE_ID}_{NOISE_TAG_SPATIAL}.npy")
    
    vz = np.load(vz_path)
    vx = np.load(vx_path)
    
    div_path = os.path.join(SPATIAL_DIR, "div", f"div_{SAMPLE_ID}_clean.npy")
    curl_path = os.path.join(SPATIAL_DIR, "curl", f"curl_{SAMPLE_ID}_clean.npy")
    true_p = np.load(div_path) if os.path.exists(div_path) else None
    true_s = np.load(curl_path) if os.path.exists(curl_path) else None

    print(f"🔄 正在对样本 {SAMPLE_ID} 进行 FK 扇形视速度滤波分离...")
    p_sep, s_sep = fk_fan_filter_separation(vz, vx, dt=DT, dx=DX)
    print("✅ 分离完成！")

    # ==================== 新增：保存预测结果为 .npy 格式 ====================
    p_save_path = f"./p_est_{SAMPLE_ID}_noise{NOISE_LEVEL}.npy"
    s_save_path = f"./s_est_{SAMPLE_ID}_noise{NOISE_LEVEL}.npy"
    
    np.save(p_save_path, p_sep)
    np.save(s_save_path, s_sep)
    print(f"💾 P波预测结果已保存至: {p_save_path}")
    print(f"💾 S波预测结果已保存至: {s_save_path}")
    # ======================================================================

    # 优化布局尺寸：采用更紧凑的横向比例，避免拉得过长或过窄
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.5), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.15, hspace=0.2, left=0.08, right=0.98, bottom=0.12, top=0.92)

    vmax_global = max(np.max(np.abs(vz)), np.max(np.abs(vx)))
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax_global, vmax=vmax_global)

    plots = [
        (axes[0, 0], vz, r"(a) Input $V_z$"),
        (axes[0, 1], p_sep, r"(c) FK-Fan P"),
        (axes[0, 2], true_p if true_p is not None else np.zeros_like(p_sep), r"(e) True P"),
        (axes[1, 0], vx, r"(b) Input $V_x$"),
        (axes[1, 1], s_sep, r"(d) FK-Fan S"),
        (axes[1, 2], true_s if true_s is not None else np.zeros_like(s_sep), r"(f) True S")
    ]

    for ax, data, title in plots:
        im = ax.imshow(
            data, cmap="seismic", aspect="auto", norm=norm,
            extent=[DEPTH_AXIS[0], DEPTH_AXIS[-1], TIME_AXIS[-1], TIME_AXIS[0]]
        )
        ax.set_title(title, loc='left', pad=3, fontsize=7)
        ax.set_xticks([DEPTH_START, (DEPTH_START+DEPTH_END)/2, DEPTH_END])
        ax.set_xticklabels([f"{DEPTH_START:.0f}", f"{(DEPTH_START+DEPTH_END)/2:.0f}", f"{DEPTH_END:.0f}"], rotation=15)
        ax.set_yticklabels(["0.0", "0.5", "1.0", "1.5", "2.0"])
        
        # 只在最左侧和最底部保留轴标签，减少视觉拥挤
        if ax in axes[:, 0]:
            ax.set_ylabel("Time (s)")
        if ax in axes[1, :]:
            ax.set_xlabel("Depth (m)")

    plt.savefig(SAVE_BASENAME + ".pdf")
    plt.savefig(SAVE_BASENAME + ".png")
    plt.close()
    print(f"📊 优化的结果图已保存至: {SAVE_BASENAME}.pdf / .png")