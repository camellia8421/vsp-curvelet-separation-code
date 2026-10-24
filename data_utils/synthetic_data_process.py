import os
import numpy as np
import pyct
import cv2
from tqdm import tqdm

# ============================================================
# 全局对齐参数
# ============================================================
TARGET_SHAPE = (512, 64)    # (时间点数, 深度道数)
TARGET_DT = 0.004           # 目标时间采样间隔：4ms
DEPTH_START = 1110.94       # 目标深度起始：套管噪声分界点
DEPTH_END = 2370.94         # 目标深度终止：1110.94 + (64-1)*20 = 2370.94m

# 原始合成数据固定参数
RAW_DT = 0.002              # 原始时间采样间隔：2ms
RAW_DEPTH_START = 50        # 原始深度起始：50m
RAW_DEPTH_INTERVAL = 20     # 原始深度间隔：20m
RAW_DEPTH_NUM = 126         # 原始深度道数：126
RAW_TIME_NUM = 2048         # 原始时间点数：2048


def curvelet_transform(raw_data_root, output_root, spatial_output_root=None):
    # 曲波变换配置（适配目标尺寸）
    nbs = 3
    nba = 8
    fdct2_obj = pyct.fdct2(
        n=TARGET_SHAPE,
        nbs=nbs,
        nba=nba,
        ac=True,
        vec=True,
        cpx=False
    )
    
    components = ["vx", "vz", "div", "curl"]
    target_num_subbands = None
    success_count = 0

    # 预生成深度坐标（避免循环内重复计算）
    target_depth_coords = np.linspace(DEPTH_START, DEPTH_END, TARGET_SHAPE[1])
    raw_depth_coords = RAW_DEPTH_START + np.arange(RAW_DEPTH_NUM) * RAW_DEPTH_INTERVAL

    for comp in components:
        comp_out_dir = os.path.join(output_root, comp)
        os.makedirs(comp_out_dir, exist_ok=True)

        # ========== 新增：创建时空域数据保存目录 ==========
        if spatial_output_root is not None:
            spatial_comp_dir = os.path.join(spatial_output_root, comp)
            os.makedirs(spatial_comp_dir, exist_ok=True)
        # ==================================================

        comp_raw_dir = os.path.join(raw_data_root, comp)
        raw_files = [f for f in os.listdir(comp_raw_dir) if f.endswith(".npy")]
        
        if not raw_files:
            print(f"⚠️ 警告：{comp}文件夹无.npy文件，已跳过")
            continue
        
        print(f"\n===== 开始处理{comp}分量（共{len(raw_files)}个文件）=====")
        for file in tqdm(raw_files, desc=f"处理{comp}"):
            try:
                raw_path = os.path.join(comp_raw_dir, file)
                data = np.load(raw_path)
                
                # 校验原始维度（已修正为2048×126）
                if data.shape != (RAW_TIME_NUM, RAW_DEPTH_NUM):
                    print(f"❌ 跳过{file}：原始维度不符（实际{data.shape}，预期({RAW_TIME_NUM}, {RAW_DEPTH_NUM})）")
                    continue

                # --------------------------
                # 1. 时间维度对齐：2ms → 4ms，截取前512个目标点
                # --------------------------
                target_total_time = TARGET_SHAPE[0] * TARGET_DT
                needed_raw_samples = int(np.round(target_total_time / RAW_DT))  # 1024个原始点
                
                # 截取前1024个原始时间点（对应前2.048s）
                if data.shape[0] >= needed_raw_samples:
                    data_time_cut = data[:needed_raw_samples, :].astype(np.float32)
                else:
                    data_time_cut = np.zeros((needed_raw_samples, data.shape[1]), dtype=np.float32)
                    data_time_cut[:data.shape[0], :] = data.astype(np.float32)
                
                # 2倍下采样：2ms间隔 → 4ms间隔
                downsample_factor = int(np.round(TARGET_DT / RAW_DT))
                data_time_down = data_time_cut[::downsample_factor, :]
                
                # 精确裁剪到目标时间点数（512点）
                if data_time_down.shape[0] > TARGET_SHAPE[0]:
                    data_time_down = data_time_down[:TARGET_SHAPE[0], :]
                elif data_time_down.shape[0] < TARGET_SHAPE[0]:
                    pad = np.zeros((TARGET_SHAPE[0] - data_time_down.shape[0], data_time_down.shape[1]), dtype=np.float32)
                    data_time_down = np.vstack([data_time_down, pad])

                # --------------------------
                # 2. 深度维度对齐：截取有效井段，插值到64道
                # --------------------------
                data_depth_aligned = np.zeros((TARGET_SHAPE[0], TARGET_SHAPE[1]), dtype=np.float32)
                for t_idx in range(TARGET_SHAPE[0]):
                    data_depth_aligned[t_idx, :] = np.interp(
                        target_depth_coords, raw_depth_coords, data_time_down[t_idx, :],
                        left=0, right=0
                    )
                
                # 双线性插值精修，保证尺寸严格匹配
                data_used = cv2.resize(
                    data_depth_aligned,
                    (TARGET_SHAPE[1], TARGET_SHAPE[0]),
                    interpolation=cv2.INTER_LINEAR
                )

                if data_used.shape != TARGET_SHAPE:
                    print(f"❌ 跳过{file}：调整后维度不符（实际{data_used.shape}，预期{TARGET_SHAPE}）")
                    continue
                
                # ========== 新增：保存裁剪对齐后的时空域原始数据 ==========
                if spatial_output_root is not None:
                    spatial_save_path = os.path.join(spatial_comp_dir, file)
                    np.save(spatial_save_path, data_used)
                # ========================================================
                
                # --------------------------
                # 3. 曲波正变换
                # --------------------------
                clarray = fdct2_obj.fwd(data_used)
                cl_struct = fdct2_obj.struct(clarray)
                raw_subbands = []
                for scale in cl_struct:
                    for subband in scale:
                        raw_subbands.append(subband)
                
                current_num = len(raw_subbands)
                if target_num_subbands is None:
                    target_num_subbands = current_num
                    print(f"✅ 检测到子带总数为{target_num_subbands}，将以此为标准")
                elif current_num != target_num_subbands:
                    print(f"❌ 跳过{file}：子带数不符（实际{current_num}，标准{target_num_subbands}）")
                    continue
                
                curvelet_data = np.array(raw_subbands, dtype=object)
                output_path = os.path.join(comp_out_dir, file)
                np.save(output_path, curvelet_data)
                success_count += 1
                
            except Exception as e:
                print(f"❌ 处理{file}失败：{str(e)}")
                continue
    
    print(f"\n===== 处理完成 =====")
    print(f"总成功处理文件数：{success_count}")
    print(f"曲波变换输入尺寸：{TARGET_SHAPE}")
    print(f"时间范围：0 ~ {TARGET_SHAPE[0]*TARGET_DT:.3f}s，采样间隔{TARGET_DT*1000:.0f}ms")
    print(f"深度范围：{DEPTH_START:.2f} ~ {DEPTH_END:.2f}m，采样间隔20m")
    print(f"子带总数标准：{target_num_subbands}")
    print(f"曲波系数结果保存至：{output_root}")
    if spatial_output_root is not None:
        print(f"时空域对齐数据保存至：{spatial_output_root}")


if __name__ == "__main__":
    RAW_DATA_DIR = "./data/train_model_62"
    OUTPUT_DIR = "./data/train_model_62_raw_subbands_512x64"
    SPATIAL_OUTPUT_DIR = "./data/train_model_preprocessed_512x64"
    
    # 执行曲波变换
    curvelet_transform(RAW_DATA_DIR, OUTPUT_DIR, SPATIAL_OUTPUT_DIR)
    
    # 验证生成文件格式（以curl分量为例）
    comp = "curl"
    comp_out_dir = os.path.join(OUTPUT_DIR, comp)
    raw_files = [f for f in os.listdir(comp_out_dir) if f.endswith(".npy")]
    
    if raw_files:
        test_path = os.path.join(comp_out_dir, raw_files[0])
        curvelet_data = np.load(test_path, allow_pickle=True)
        
        print("\n===== 曲波系数格式验证结果 =====")
        print("保存的数组形状：", curvelet_data.shape)
        print("保存的数组 dtype：", curvelet_data.dtype)
        print("第一个子带的形状：", curvelet_data[0].shape)
        print("第一个子带的数据类型：", curvelet_data[0].dtype)
        print("是否包含复数：", np.iscomplexobj(curvelet_data[0]))

        # 新增：验证时空域数据格式
        spatial_test_path = os.path.join(SPATIAL_OUTPUT_DIR, comp, raw_files[0])
        if os.path.exists(spatial_test_path):
            spatial_data = np.load(spatial_test_path)
            print("\n===== 时空域对齐数据格式验证结果 =====")
            print("数据形状：", spatial_data.shape)
            print("数据 dtype：", spatial_data.dtype)
            print("数值范围：", spatial_data.min(), "~", spatial_data.max())
    else:
        print(f"⚠️ 未找到{comp}分量的.npy文件，无法进行格式验证")
