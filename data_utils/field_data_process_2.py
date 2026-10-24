import os
import numpy as np
import matplotlib.pyplot as plt
import pyct
from scipy.signal import butter, filtfilt
# ----------------配置----------------
N_TIME = 512
N_DEPTH = 64
DT = 0.004            # 时间采样(s)
LOWPASS_CUTOFF = 120  # <125，小于奈奎斯特250/2=125Hz

# 球面增益配置
APPLY_SPHERICAL = True
T0 = 0.02
SPHERE_POW = 0.75
MAX_TIME_GAIN = 8.0

# ✨斜向初至截断参数
CUT_T_SHALLOW = 130    # 最浅道(depth=0)硬截断时间索引
CUT_T_DEEP    = 190    # 最深道(depth=63)硬截断时间索引
TAPER_LENGTH  = 20     # 余弦斜坡过渡采样点数

vz_npy_path = "./vz_shifted_512x64.npy"
vx_npy_path = "./vx_shifted_512x64.npy"

# 空间域预处理结果保存目录
OUTPUT_PROCESSED = "./data/real_vsp_processed/"
os.makedirs(OUTPUT_PROCESSED, exist_ok=True)
# 曲波子带输出目录
OUTPUT_CURVELET = "./data/real_vsp_curvelet_64_512"
os.makedirs(OUTPUT_CURVELET, exist_ok=True)
FIG_DIR = os.path.join(OUTPUT_CURVELET, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

CURVELET_PARAMS = dict(
    n=(N_TIME, N_DEPTH),
    nbs=3,
    nba=8,
    ac=True,
    vec=True,
    cpx=False
)


def normalize_global(arr):
    """全局归一化到[-1,1]，仅绘图"""
    amp = np.max(np.abs(arr))
    if amp < 1e-9:
        return arr
    return arr / amp


def spherical_amplitude_correct(data, dt, t0=0.02, gain_pow=1.0, max_gain=8.0):
    """球面扩散补偿，增益截断防噪声爆炸"""
    nt, nd = data.shape
    t_vec = np.arange(nt) * dt
    t_safe = np.maximum(t_vec, t0)
    gain = t_safe ** gain_pow
    gain = np.clip(gain, None, max_gain)
    gain = gain[:, np.newaxis]
    return data * gain


def trace_balance(arr):
    """按道峰值归一"""
    nt, nd = arr.shape
    out = np.zeros_like(arr, dtype=np.float32)
    for ich in range(nd):
        tr = arr[:, ich]
        max_amp = np.max(np.abs(tr))
        if max_amp > 1e-9:
            out[:, ich] = tr / max_amp
        else:
            out[:, ich] = tr
    return out


def time_lowpass(arr, dt, cutoff):
    """时间维度低通滤波 axis=0 时间轴"""
    fs = 1.0 / dt
    b, a = butter(4, cutoff, fs=fs, btype='lowpass')
    return filtfilt(b, a, arr, axis=0)


if __name__ == "__main__":
    vz = np.load(vz_npy_path)
    vx = np.load(vx_npy_path)
    print(f"读入原始数据 shape | vz:{vz.shape}, vx:{vx.shape}")
    assert vz.shape == (N_TIME, N_DEPTH)
    assert vx.shape == (N_TIME, N_DEPTH)

    # ==========流水线：球面增益 →低通滤波 →按道归一 ==========
    if APPLY_SPHERICAL:
        vz_gain = spherical_amplitude_correct(vz, dt=DT, t0=T0, gain_pow=SPHERE_POW, max_gain=MAX_TIME_GAIN)
        vx_gain = spherical_amplitude_correct(vx, dt=DT, t0=T0, gain_pow=SPHERE_POW, max_gain=MAX_TIME_GAIN)
        print(f"✅球面扩散增益开启 pow={SPHERE_POW}, max_gain={MAX_TIME_GAIN}")
    else:
        vz_gain = vz
        vx_gain = vx

    vz_lp = time_lowpass(vz_gain, dt=DT, cutoff=LOWPASS_CUTOFF)
    vx_lp = time_lowpass(vx_gain, dt=DT, cutoff=LOWPASS_CUTOFF)
    print(f"✅时间低通滤波完成，cutoff={LOWPASS_CUTOFF} Hz")

    vz_bal = trace_balance(vz_lp)
    vx_bal = trace_balance(vx_lp)
    print(f"✅按道归一化(trace balance)完成")

    # ===================== ✨新增：斜向硬置零 + 余弦taper截断 =====================
    nt, nd = vz_bal.shape
    cut_t_per_trace = np.linspace(CUT_T_SHALLOW, CUT_T_DEEP, num=nd, dtype=int)

    for ich in range(nd):
        t_cut = cut_t_per_trace[ich]
        taper = np.ones(nt, dtype=np.float32)
        # t_cut 之前全部强制置0
        taper[:t_cut] = 0.0
        # t_cut ~ t_cut+TAPER_LENGTH 余弦斜坡从0升到1
        ramp_end = t_cut + TAPER_LENGTH
        if ramp_end > nt:
            ramp_end = nt
        n_ramp = ramp_end - t_cut
        if n_ramp > 0:
            taper[t_cut:ramp_end] = np.cos(np.linspace(np.pi, 0, n_ramp)) * 0.5 + 0.5

        vz_bal[:, ich] *= taper
        vx_bal[:, ich] *= taper

    print(f"✅斜向截断完成：depth0→t={CUT_T_SHALLOW}, depth{nd-1}→t={CUT_T_DEEP}, taper_len={TAPER_LENGTH}")
    print(f"每道截断时间范围 {cut_t_per_trace.min()} ~ {cut_t_per_trace.max()}")
    # ==========================================================================

    # ----------------------保存空间域数据到 /home/litianhao/data/real_vsp_processed/ ----------------------
    np.save(os.path.join(OUTPUT_PROCESSED, "vz_spherical_gain.npy"), vz_gain)
    np.save(os.path.join(OUTPUT_PROCESSED, "vx_spherical_gain.npy"), vx_gain)
    np.save(os.path.join(OUTPUT_PROCESSED, "vz_lowpass.npy"), vz_lp)
    np.save(os.path.join(OUTPUT_PROCESSED, "vx_lowpass.npy"), vx_lp)
    np.save(os.path.join(OUTPUT_PROCESSED, "vz_processed.npy"), vz_bal)
    np.save(os.path.join(OUTPUT_PROCESSED, "vx_processed.npy"), vx_bal)
    print(f"\n✅空间域预处理文件已保存至 {OUTPUT_PROCESSED}")

    # 绘图，叠加红色截断斜线
    vz_norm = normalize_global(vz_bal)
    vx_norm = normalize_global(vx_bal)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axes[0].imshow(vz_norm, cmap="seismic", vmin=-1, vmax=1, aspect="auto", origin="upper")
    axes[0].set_title("vz (spherical+lowpass + trace‑balanced + slant‑cut)")
    axes[0].set_xlabel("Depth index")
    axes[0].set_ylabel("Time index")
    axes[0].plot(np.arange(nd), cut_t_per_trace, "r-", lw=1.5)   #红色参考线
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(vx_norm, cmap="seismic", vmin=-1, vmax=1, aspect="auto", origin="upper")
    axes[1].set_title("vx (spherical+lowpass + trace‑balanced + slant‑cut)")
    axes[1].set_xlabel("Depth index")
    axes[1].set_ylabel("Time index")
    axes[1].plot(np.arange(nd), cut_t_per_trace, "r-", lw=1.5)   #红色参考线
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    raw_fig_path = os.path.join(FIG_DIR, "preprocessed_vz_vx.png")
    plt.savefig(raw_fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"预处理后vz/vx图保存: {raw_fig_path}")

    # 曲波正变换
    fdct2_obj = pyct.fdct2(**CURVELET_PARAMS)
    cl_vz = fdct2_obj.fwd(vz_bal)
    struct_vz = fdct2_obj.struct(cl_vz)
    subbands_vz = []
    for scale in struct_vz:
        for sb in scale:
            subbands_vz.append(sb)
    np.save(os.path.join(OUTPUT_CURVELET, "vz_preprocessed.npy"), np.array(subbands_vz, dtype=object))

    cl_vx = fdct2_obj.fwd(vx_bal)
    struct_vx = fdct2_obj.struct(cl_vx)
    subbands_vx = []
    for scale in struct_vx:
        for sb in scale:
            subbands_vx.append(sb)
    np.save(os.path.join(OUTPUT_CURVELET, "vx_preprocessed.npy"), np.array(subbands_vx, dtype=object))

    # 绘制vz全部子带
    print(f"\n开始绘制vz曲波子带，总子带数：{len(subbands_vz)}")
    for idx, sb_data in enumerate(subbands_vz):
        sb_norm = normalize_global(sb_data)
        plt.figure(figsize=(8, 6))
        plt.imshow(sb_norm, cmap="seismic", vmin=-1, vmax=1, aspect="auto", origin="upper")
        plt.title(f"Curvelet subband vz #{idx}, shape={sb_data.shape}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar(shrink=0.8)
        plt.tight_layout()
        sb_fig_path = os.path.join(FIG_DIR, f"vz_subband_{idx:02d}.png")
        plt.savefig(sb_fig_path, dpi=120, bbox_inches="tight")
        plt.close()
    print(f"vz所有子带图片保存至 {FIG_DIR}")

    # 输出校验
    test_vz = np.load(os.path.join(OUTPUT_CURVELET, "vz_preprocessed.npy"), allow_pickle=True)
    test_vx = np.load(os.path.join(OUTPUT_CURVELET, "vx_preprocessed.npy"), allow_pickle=True)
    print("\n====输出校验====")
    print(f"vz子带数量:{len(test_vz)}")
    print(f"vx子带数量:{len(test_vx)}")
    print(f"第一个子带shape:{test_vz[0].shape}")
    print(f"\n✅空间域结果:{OUTPUT_PROCESSED}")
    print(f"✅曲波子带结果:{OUTPUT_CURVELET}")
