import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftfreq
from scipy.signal import tukey
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from matplotlib.colors import TwoSlopeNorm
import warnings

# ====================== 全局配置 ======================
# 真实VSP数据路径
PATH_VZ = "./data/real_vsp_processed/vz_processed.npy"
PATH_VX = "./data/real_vsp_processed/vx_processed.npy"

# 输出保存路径
SAVE_OUT_DIR = "./fk_vsp_output"

# 物理参数
DT = 0.004               # 时间采样间隔 (s)
DEPTH_START = 674.72
DEPTH_END = 1934.72

# 波场分离核心参数（请根据真实地层速度调整）
VP = 2500.0              # P波速度 (m/s)
VS = 1443.0              # S波速度 (m/s)，默认VP/√3≈1443
FMAX = 45.0              # 最高有效频率 (Hz)
TAPER = 0.25             # Tukey窗系数

# 绘图配置
FIG_CM_WIDTH = 16
FIG_CM_HEIGHT = 8
# ======================================================

# 创建输出目录
os.makedirs(SAVE_OUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def vector_fk_polarization_separate(vx, vz, dt, dx, vp, vs, f_max=45.0, taper=0.25):
    """
    矢量FK域偏振分离算法（与原代码完全一致）
    输入：水平分量vx，垂直分量vz，时间间隔dt，空间间隔dx，P波速度vp，S波速度vs
    输出：P波水平分量p_x、P波垂直分量p_z、S波水平分量s_x、S波垂直分量s_z
    """
    nt, nx = vx.shape
    
    # 时间域加Tukey窗抑制边缘伪影
    win_t = tukey(nt, taper)[:, None]
    vx_tap = vx * win_t
    vz_tap = vz * win_t
    
    # 二维FFT并频域中心化
    Fvx = fftshift(fft2(vx_tap))
    Fvz = fftshift(fft2(vz_tap))
    freqs = fftshift(fftfreq(nt, dt))
    kx = fftshift(fftfreq(nx, dx))
    
    # 构造频率-波数网格
    F, KX = np.meshgrid(freqs, kx, indexing='ij')
    
    # 计算P波、S波的波数标量
    k_p = np.abs(F) / (vp + 1e-12)
    k_s = np.abs(F) / (vs + 1e-12)
    
    # 计算垂直波数kz（仅保留传播波，衰减波置零）
    kz_p = np.zeros_like(KX)
    mask_p = (k_p**2 > KX**2)
    kz_p[mask_p] = np.sqrt(k_p[mask_p]**2 - KX[mask_p]**2)
    
    kz_s = np.zeros_like(KX)
    mask_s = (k_s**2 > KX**2)
    kz_s[mask_s] = np.sqrt(k_s[mask_s]**2 - KX[mask_s]**2)
    
    # 矢量投影分解：P波纵波偏振、S波横波偏振
    denom_p = (KX**2 + kz_p**2 + 1e-12)
    proj_p = (KX * Fvx + kz_p * Fvz) / denom_p
    
    denom_s = (kz_s**2 + KX**2 + 1e-12)
    proj_s = (-kz_s * Fvx + KX * Fvz) / denom_s
    
    # 重建各分量的FK谱
    Fvx_p = proj_p * KX
    Fvz_p = proj_p * kz_p
    Fvx_s = proj_s * (-kz_s)
    Fvz_s = proj_s * KX
    
    # 仅保留上行波（kz>0 对应向上传播）
    up_mask_p = kz_p > 1e-8
    up_mask_s = kz_s > 1e-8
    Fvx_p *= up_mask_p
    Fvz_p *= up_mask_p
    Fvx_s *= up_mask_s
    Fvz_s *= up_mask_s
    
    # 偏振置信度加权，抑制偏振不明确区域的噪声
    total_energy = Fvx**2 + Fvz**2 + 1e-12
    energy_p = Fvx_p**2 + Fvz_p**2
    energy_s = Fvx_s**2 + Fvz_s**2
    conf_p = np.clip(energy_p / total_energy, 0, 1)
    conf_s = np.clip(energy_s / total_energy, 0, 1)
    Fvx_p *= conf_p
    Fvz_p *= conf_p
    Fvx_s *= conf_s
    Fvz_s *= conf_s
    
    # 反变换回时空域，取实部
    p_x = np.real(ifft2(ifftshift(Fvx_p)))
    p_z = np.real(ifft2(ifftshift(Fvz_p)))
    s_x = np.real(ifft2(ifftshift(Fvx_s)))
    s_z = np.real(ifft2(ifftshift(Fvz_s)))
    
    return p_x, p_z, s_x, s_z


if __name__ == "__main__":
    # 1. 读取真实VSP数据
    if not os.path.exists(PATH_VZ) or not os.path.exists(PATH_VX):
        raise FileNotFoundError(f"未找到数据文件，请检查路径：\n{PATH_VZ}\n{PATH_VX}")
    
    vz = np.load(PATH_VZ)
    vx = np.load(PATH_VX)
    nt, nr = vz.shape
    print(f"📥 读取真实VSP数据完成：{nt} 时间点 × {nr} 深度道")
    
    # 2. 动态计算物理参数与坐标轴
    DX = (DEPTH_END - DEPTH_START) / (nr - 1)
    TIME_AXIS  = np.linspace(0, nt * DT, nt)
    DEPTH_AXIS = np.linspace(DEPTH_START, DEPTH_END, nr)
    print(f"📐 道间距 DX = {DX:.2f} m")
    
    # 3. 矢量FK偏振分离
    p_x, p_z, s_x, s_z = vector_fk_polarization_separate(
        vx, vz, DT, DX, VP, VS, FMAX, TAPER
    )

    p_pred = p_z
    s_pred = s_x
    print("✅ 矢量FK偏振分离完成")
    
    # 4. 保存分离后的P/S波npy文件
    np.save(os.path.join(SAVE_OUT_DIR, "real_vsp_P_pred.npy"), p_pred)
    np.save(os.path.join(SAVE_OUT_DIR, "real_vsp_S_pred.npy"), s_pred)
    print(f"💾 分离结果已保存至 {SAVE_OUT_DIR}/ 目录：")
    print(f"   - real_vsp_P_pred.npy (P波，垂直分量)")
    print(f"   - real_vsp_S_pred.npy (S波，水平分量)")
    
    # 5. 绘图展示（原始数据 + 分离结果对比）
    # 全局振幅归一化，视觉统一
    all_data = [vz, p_pred, vx, s_pred]
    vmax_global = np.max([np.abs(arr).max() for arr in all_data])
    norm_global = TwoSlopeNorm(vcenter=0, vmin=-vmax_global, vmax=vmax_global)
    
    fig, axes = plt.subplots(2, 2, figsize=(FIG_CM_WIDTH/2.54, FIG_CM_HEIGHT/2.54))
    plt.subplots_adjust(wspace=0.22, hspace=0.32)
    
    extent = [DEPTH_START, DEPTH_END, TIME_AXIS[-1], TIME_AXIS[0]]
    
    # 第一行：垂直分量
    axes[0,0].imshow(vz, cmap='seismic', norm=norm_global, aspect='auto', extent=extent)
    axes[0,0].set_title("Original $V_z$")
    
    axes[0,1].imshow(p_pred, cmap='seismic', norm=norm_global, aspect='auto', extent=extent)
    axes[0,1].set_title("Vector-FK Separated P-wave")
    
    # 第二行：水平分量
    axes[1,0].imshow(vx, cmap='seismic', norm=norm_global, aspect='auto', extent=extent)
    axes[1,0].set_title("Original $V_x$")
    
    axes[1,1].imshow(s_pred, cmap='seismic', norm=norm_global, aspect='auto', extent=extent)
    axes[1,1].set_title("Vector-FK Separated S-wave")
    
    # 坐标轴规范（与原代码完全一致）
    for idx, ax in enumerate(axes.flat):
        ax.set_xticks([DEPTH_START, (DEPTH_START+DEPTH_END)/2, DEPTH_END])
        ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
        if idx % 2 == 0:
            ax.set_xticklabels([f"{DEPTH_START:.0f}", f"{(DEPTH_START+DEPTH_END)/2:.0f}", f"{DEPTH_END:.0f}"])
            ax.set_xlabel("Depth (m)")
            ax.set_ylabel("Time (s)")
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    
    # 统一色条
    cbar = fig.colorbar(axes[0,0].images[0], ax=axes, shrink=0.86, pad=0.02, aspect=28)
    cbar.set_label("Amplitude", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    
    plt.savefig("VectorFK_Real_VSP.png")
    plt.savefig("VectorFK_Real_VSP.pdf")
    plt.close()
    print("📊 绘图完成，已输出 PNG/PDF 格式")
