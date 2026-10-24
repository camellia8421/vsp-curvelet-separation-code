import os
import numpy as np
import segyio
import pandas as pd
import matplotlib.pyplot as plt

SEGY_PATH = "./data/field_data/seam_vsp/TR_McMillen2_2D_VSP_20190403_rotated_shot_recs_with_geom_GeostockSandia.sgy"
GEOM_PATH = "./data/field_data/seam_vsp/TR_McMillen2_2D_VSP_2019_rcv_geometry_GeostockSandia.txt"
SEG_CSV = "./data/field_data/seam_vsp/shot_segment_list.txt"

DEEP_FLDR = 4
MID_FLDR = 601
TARGET_DEPTH_END = 1934.72
N_DEPTH = 64
DEPTH_STEP = 20
DT_NEW = 0.004
N_TIME = 512

# 目标初至平移到0.5s，计算平移采样点数
TARGET_FIRST_ARRIVAL = 0.35
shift_samples = int(TARGET_FIRST_ARRIVAL / DT_NEW)
print(f"时间平移采样点: {shift_samples}, 对应时间 {shift_samples*DT_NEW:.2f} s")

def read_geom(geom_path):
    pod_nums, xs, ys, depths_ft = [], [], [], []
    with open(geom_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Receiver') or line.startswith('pod'):
                continue
            parts = line.split()
            pod_nums.append(int(float(parts[0])))
            xs.append(float(parts[1]))
            ys.append(float(parts[2]))
            depths_ft.append(float(parts[3]))
    return np.array(pod_nums), np.array(xs), np.array(ys), np.array(depths_ft) * 0.3048

def read_shot_segment(segy_path, trace_beg, trace_end):
    with segyio.open(segy_path, ignore_geometry=True) as f:
        dt = f.bin[segyio.BinField.Interval] / 1_000_000.0
        n_samp = f.samples.size
        n_trace = int(trace_end - trace_beg + 1)
        n_group = n_trace // 3
        R = np.array([f.trace[int(trace_beg) + g*3]     for g in range(n_group)])
        T = np.array([f.trace[int(trace_beg) + g*3 + 1] for g in range(n_group)])
        Z = np.array([f.trace[int(trace_beg) + g*3 + 2] for g in range(n_group)])
    return R, T, Z, dt, n_samp

def get_segment_depths(segy_path, geom_path, trace_beg, n_group):
    pod_nums, geom_x, geom_y, depths_m = read_geom(geom_path)
    depths = []
    with segyio.open(segy_path, ignore_geometry=True) as f:
        for g in range(n_group):
            h = f.header[int(trace_beg) + g*3]
            gx = h[segyio.TraceField.GroupX] / 1000.0
            gy = h[segyio.TraceField.GroupY] / 1000.0
            dist = np.sqrt((geom_x - gx)**2 + (geom_y - gy)**2)
            idx = np.argmin(dist)
            depths.append(depths_m[idx])
    return np.array(depths)

def interp_time(data, orig_time, new_time):
    n_rec = data.shape[0]
    out = np.zeros((n_rec, len(new_time)))
    for i in range(n_rec):
        out[i,:] = np.interp(new_time, orig_time, data[i,:])
    return out

def interp_depth(data, orig_depth, new_depth):
    n_t = data.shape[1]
    out = np.zeros((len(new_depth), n_t))
    for j in range(n_t):
        out[:,j] = np.interp(new_depth, orig_depth, data[:,j])
    return out

def shift_time(data, shift_samp):
    """时间轴后移shift_samp个点，开头补0，末尾截断"""
    n_rec, n_samp = data.shape
    out = np.zeros_like(data)
    if shift_samp >= n_samp:
        return out
    out[:, shift_samp:] = data[:, :-shift_samp]
    return out

def plot_save(R, T, Z, depth_axis, dt, fn, title):
    n_d, n_t = R.shape
    t_axis = np.arange(n_t)*dt
    def bal(arr):
        amp = np.max(np.abs(arr), axis=1, keepdims=True)
        amp[amp<1e-9] = 1.0
        return arr/amp
    Rb=bal(R);Tb=bal(T);Zb=bal(Z)
    fig,axes=plt.subplots(1,3,figsize=(16,7),sharey=True)
    ext=[depth_axis.min(),depth_axis.max(), t_axis.max(), t_axis.min()]
    im0=axes[0].imshow(Rb.T,extent=ext,origin="upper",cmap="seismic",vmin=-1,vmax=1,aspect="auto")
    axes[0].set_title(f"Radial | {title}")
    axes[0].set_xlabel("Depth(m)");axes[0].set_ylabel("Time(s)")
    plt.colorbar(im0,ax=axes[0],shrink=0.7)
    im1=axes[1].imshow(Tb.T,extent=ext,origin="upper",cmap="seismic",vmin=-1,vmax=1,aspect="auto")
    axes[1].set_title(f"Transverse | {title}")
    axes[1].set_xlabel("Depth(m)")
    plt.colorbar(im1,ax=axes[1],shrink=0.7)
    im2=axes[2].imshow(Zb.T,extent=ext,origin="upper",cmap="seismic",vmin=-1,vmax=1,aspect="auto")
    axes[2].set_title(f"Vertical | {title}")
    axes[2].set_xlabel("Depth(m)")
    plt.colorbar(im2,ax=axes[2],shrink=0.7)
    plt.tight_layout()
    plt.savefig(fn,dpi=150,bbox_inches="tight")
    plt.show()
    print(f"图片保存: {fn}")

if __name__ == "__main__":
    df = pd.read_csv(SEG_CSV)
    seg_deep = df[df["shot_no"] == DEEP_FLDR].iloc[0]
    seg_mid  = df[df["shot_no"] == MID_FLDR].iloc[0]

    Rd, Td, Zd, dt_deep, ns_deep = read_shot_segment(SEGY_PATH, seg_deep.trace_beg, seg_deep.trace_end)
    dep_deep = get_segment_depths(SEGY_PATH, GEOM_PATH, seg_deep.trace_beg, int(seg_deep.n_group))

    Rm, Tm, Zm, dt_mid, ns_mid = read_shot_segment(SEGY_PATH, seg_mid.trace_beg, seg_mid.trace_end)
    dep_mid = get_segment_depths(SEGY_PATH, GEOM_PATH, seg_mid.trace_beg, int(seg_mid.n_group))

    print(f"中部串深度 {dep_mid.min():.2f} ~ {dep_mid.max():.2f} m")
    print(f"深部串深度 {dep_deep.min():.2f} ~ {dep_deep.max():.2f} m")

    t_new = np.arange(N_TIME) * DT_NEW
    t_orig_deep = np.arange(ns_deep)*dt_deep
    t_orig_mid = np.arange(ns_mid)*dt_mid

    # 时间插值
    Rd_t = interp_time(Rd, t_orig_deep, t_new)
    Td_t = interp_time(Td, t_orig_deep, t_new)
    Zd_t = interp_time(Zd, t_orig_deep, t_new)

    Rm_t = interp_time(Rm, t_orig_mid, t_new)
    Tm_t = interp_time(Tm, t_orig_mid, t_new)
    Zm_t = interp_time(Zm, t_orig_mid, t_new)

    # 合并时间插值后波形
    orig_depth_all = np.concatenate([dep_mid, dep_deep])
    R_tcat = np.concatenate([Rm_t, Rd_t], axis=0)
    T_tcat = np.concatenate([Tm_t, Td_t], axis=0)
    Z_tcat = np.concatenate([Zm_t, Zd_t], axis=0)

    # 深度网格，终点固定1934.72
    d_max = TARGET_DEPTH_END
    d_min = d_max - (N_DEPTH-1)*DEPTH_STEP
    depths_new = np.linspace(d_min, d_max, N_DEPTH)

    # 深度插值
    R_interp = interp_depth(R_tcat, orig_depth_all, depths_new)
    T_interp = interp_depth(T_tcat, orig_depth_all, depths_new)
    Z_interp = interp_depth(Z_tcat, orig_depth_all, depths_new)

    # ---------------- 全局时间后移，初至移到0.5s ----------------
    R_shifted = shift_time(R_interp, shift_samples)
    T_shifted = shift_time(T_interp, shift_samples)
    Z_shifted = shift_time(Z_interp, shift_samples)

    plot_save(R_shifted, T_shifted, Z_shifted, depths_new, DT_NEW,
              "./interp_time_shifted_0.5s.png",
              f"shift {shift_samples} samp, first arrival ~0.5s")

    # 输出数组格式 (time, depth)，和你网络输入对齐
    vz_out = Z_shifted.T
    vx_out = R_shifted.T
    print(f"输出数组形状 vz:{vz_out.shape}, vx:{vx_out.shape}")

    # 保存平移之后的npy
    np.save("./vz_shifted_512x64.npy", vz_out)
    np.save("./vx_shifted_512x64.npy", vx_out)
    print("已保存平移后数据：vz_shifted_512x64.npy , vx_shifted_512x64.npy")
