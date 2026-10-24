import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Any, Optional, Callable
import logging

# 改为模块级导入，确保动态初始化后变量能同步更新
import utils
from utils import (
    init_subband_config,
    compute_subband_norm_params, compute_subband_norm_params,
    normalize_subband
)

# ==============================
# 日志配置（统一格式，与utils保持一致）
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==============================================
# 1. 曲波域数据变换类（仅用于测试集，关闭数据增强）
# ==============================================
class CurveletAwareTransform:
    """曲波域数据变换类：仅支持随机翻转（训练时可关闭，测试时默认禁用）"""
    def __init__(self, flip_prob: float = 0.0):
        self.flip_prob = flip_prob  # 随机翻转概率（训练时设为0，无数据增强）

    def __call__(self, subband_data: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        """
        对曲波域子带数据应用变换
        修复：先创建副本，再修改副本，避免污染原始数据
        Args:
            subband_data: 格式为[[分量1子带列表], [分量2子带列表], ...]
        Returns:
            变换后的子带数据（副本）
        """
        # 先创建深拷贝，避免修改原始数据
        import copy
        subband_data = copy.deepcopy(subband_data)

        # 随机水平翻转（训练时flip_prob=0，实际无效果）
        if np.random.rand() < self.flip_prob:
            for comp_idx in range(len(subband_data)):
                for sb_idx in range(len(subband_data[comp_idx])):
                    subband_data[comp_idx][sb_idx] = np.fliplr(subband_data[comp_idx][sb_idx]).astype(np.float32)

        return subband_data

# ==============================================
# 2. 核心数据集加载类：SeismicCurveletDataloader
# ==============================================
class SeismicCurveletDataloader(Dataset):
    """
    地震曲波域子带数据集加载器（支持任意子带尺寸）
    核心特性：
    1. 输入为vx+vz双通道，输出为curl+div双通道，与模型输出维度匹配
    2. 支持子带级/全局级归一化，与训练逻辑一致
    3. 高频增强（数值缩放）在归一化前执行，保持数据分布稳定
    4. 异常值裁剪确保归一化参数计算的鲁棒性
    5. 自动重试机制处理损坏样本，提升数据加载稳定性
    6. 支持动态传入子带尺寸，自动初始化配置
    """
    def __init__(
        self,
        data_path: str,
        max_data: int = None,
        data_start: int = 0,
        jsample: int = 1,
        transform: Optional[Callable] = None,
        use_global_normalization: bool = False,    # 关键：False=子带归一化，True=全局归一化
        high_freq_enhance_factor: float = 1.0,     # 高频增强系数（仅尺度2子带，数值缩放）
        save_norm_params: bool = True,             # 是否保存归一化参数
        norm_params_path: str = None,              # 归一化参数保存/加载路径
        clip_outliers: bool = True,                # 开启异常值裁剪，确保参数计算鲁棒性
        fail_on_corrupt: bool = False,             # 遇到损坏文件是否报错
        norm_sample_count: int = 50,               # 计算归一化参数的样本数（默认前50个）
        subband_sizes: List[Tuple[int, int]] = None,  # 新增：子带尺寸列表，传入则自动初始化配置
        scale_dir_counts: List[int] = None           # 新增：每个尺度的子带数量，与subband_sizes配套
    ):
        super().__init__()
        self.data_path = data_path
        self.max_data = max_data
        self.data_start = data_start
        self.jsample = jsample
        self.transform = transform
        self.use_global_normalization = use_global_normalization  # 归一化策略开关
        self.high_freq_enhance_factor = high_freq_enhance_factor  # 高频数值增强（非噪声）
        self.save_norm_params = save_norm_params
        self.clip_outliers = clip_outliers  # 强制异常值裁剪，统一训练/测试逻辑
        self.fail_on_corrupt = fail_on_corrupt
        self.norm_sample_count = norm_sample_count  # 计算参数的样本数

        # ======================
        # 动态初始化子带配置
        # ======================
        if subband_sizes is not None:
            if scale_dir_counts is None:
                raise ValueError("传入 subband_sizes 时必须同时传入 scale_dir_counts")
            init_subband_config(subband_sizes, scale_dir_counts)

        # 校验子带配置已初始化（通过模块引用读取最新值）
        if utils.SUBBAND_COUNT == 0:
            raise RuntimeError(
                "子带配置未初始化，请先调用 utils.init_subband_config，"
                "或在数据集初始化时传入 subband_sizes 和 scale_dir_counts 参数"
            )
        logger.info(f"📐 当前子带配置：共 {utils.SUBBAND_COUNT} 个子带，尺度分布 {[len(v) for v in utils.SUBBAND_SCALE_GROUP.values()]}")

        # 归一化参数路径自动拼接尺寸标识，避免不同尺寸混用
        if norm_params_path is None:
            norm_type = 'global' if use_global_normalization else 'subband'
            # 用首个子带尺寸作为标识，区分不同配置
            size_tag = f"{utils.SUBBAND_SIZES[0][0]}x{utils.SUBBAND_SIZES[0][1]}"
            self.norm_params_path = os.path.join(data_path, f"curvelet_{norm_type}_norm_params_{size_tag}.pkl")
        else:
            self.norm_params_path = norm_params_path

        # 核心分量定义（vx + vz输入，curl + div输出）
        self.input_components = ["vx", "vz"]  # 输入双通道
        self.output_components = ["curl", "div"]  # 输出双通道
        self.all_components = self.input_components + self.output_components
        self.norm_components = self.all_components  # 归一化覆盖所有分量

        # 验证数据路径和分量目录
        self._validate_data_dirs()

        # 加载数据索引（有效样本）
        self.data_indices = self._get_valid_data_indices()
        logger.info(f"✅ 加载有效样本数：{len(self.data_indices)}（起始索引：{data_start}，最大样本：{max_data or '全部'}）")

        # 计算/加载归一化参数（覆盖所有分量）
        self.norm_params = self._load_or_compute_norm_params()
        logger.info(f"✅ 归一化参数加载完成：策略={('全局' if use_global_normalization else '子带') + '-mean_std'}，路径={self.norm_params_path}")

    def _validate_data_dirs(self) -> None:
        """验证所有分量的目录是否存在"""
        for comp in self.all_components:
            comp_dir = os.path.join(self.data_path, comp)
            if not os.path.isdir(comp_dir):
                raise FileNotFoundError(f"分量目录不存在：{comp_dir}（请检查数据路径和分量名）")
        logger.info(f"✅ 所有分量目录验证通过：{self.all_components}")

    def _get_valid_data_indices(self) -> List[int]:
        """获取有效的数据索引（所有分量都存在的样本）"""
        # 以vx分量为基准获取索引
        vx_dir = os.path.join(self.data_path, "vx")
        all_indices = sorted([
            int(f.split("_")[-1].replace(".npy", ""))
            for f in os.listdir(vx_dir)
            if f.endswith(".npy") and f.startswith("vx_")
        ])

        # 筛选起始索引和步长
        filtered_indices = all_indices[self.data_start::self.jsample]

        # 筛选所有分量都存在的索引
        valid_indices = []
        for idx in filtered_indices:
            if all(os.path.exists(os.path.join(self.data_path, comp, f"{comp}_{idx}.npy")) for comp in self.all_components):
                valid_indices.append(idx)
            elif self.fail_on_corrupt:
                raise FileNotFoundError(f"样本{idx}的部分分量文件缺失")

        # 限制最大样本数
        if self.max_data is not None and self.max_data > 0:
            valid_indices = valid_indices[:self.max_data]

        if not valid_indices:
            raise RuntimeError("无有效样本（请检查数据路径、起始索引和分量文件）")

        return valid_indices

    def _load_single_component(self, comp: str, idx: int) -> List[np.ndarray]:
        """加载单个分量的所有子带数据（包含异常值裁剪，适配动态尺寸）"""
        comp_file = os.path.join(self.data_path, comp, f"{comp}_{idx}.npy")
        try:
            subbands = np.load(comp_file, allow_pickle=True)
            # 验证子带数量（通过模块引用读取最新值）
            if len(subbands) != utils.SUBBAND_COUNT:
                raise ValueError(f"子带数量不匹配：预期{utils.SUBBAND_COUNT}，实际{len(subbands)}")
            # 转换为列表并验证尺寸（通过模块引用读取最新值）
            subbands_list = []
            for sb_idx, (sb, expected_size) in enumerate(zip(subbands, utils.SUBBAND_SIZES)):
                if sb.shape != expected_size:
                    raise ValueError(f"子带{sb_idx}尺寸不匹配：预期{expected_size}，实际{sb.shape}")
                subbands_list.append(sb.astype(np.float32))
            # 裁剪异常值（IQR方法），确保数据分布稳定
            if self.clip_outliers:
                for sb_idx in range(len(subbands_list)):
                    q1, q3 = np.percentile(subbands_list[sb_idx], [25, 75])
                    iqr = q3 - q1
                    clip_min = q1 - 1.5 * iqr
                    clip_max = q3 + 1.5 * iqr
                    subbands_list[sb_idx] = np.clip(subbands_list[sb_idx], clip_min, clip_max).astype(np.float32)
            return subbands_list
        except Exception as e:
            if self.fail_on_corrupt:
                raise RuntimeError(f"加载分量{comp}样本{idx}失败：{str(e)}") from e
            logger.warning(f"⚠️ 加载分量{comp}样本{idx}失败，跳过：{str(e)}")
            return None

    def _load_or_compute_norm_params(self) -> Dict[str, Any]:
        """加载已保存的归一化参数，若不存在则计算并保存"""
        # 尝试加载现有参数
        if os.path.exists(self.norm_params_path):
            try:
                with open(self.norm_params_path, 'rb') as f:
                    norm_params = pickle.load(f)
                # 验证参数是否匹配当前归一化策略和分量
                param_valid = True
                test_comp = self.norm_components[0]
                if self.use_global_normalization:
                    if not isinstance(norm_params.get(test_comp), tuple) or len(norm_params.get(test_comp, ())) != 2:
                        param_valid = False
                else:
                    if not isinstance(norm_params.get(test_comp), dict) or 0 not in norm_params.get(test_comp, {}):
                        param_valid = False
                if not all(comp in norm_params for comp in self.norm_components):
                    param_valid = False
                if param_valid:
                    logger.info(f"ℹ️ 加载现有归一化参数，覆盖分量：{list(norm_params.keys())}")
                    return norm_params
                else:
                    logger.warning(f"⚠️ 现有归一化参数与当前策略/分量不匹配，重新计算")
            except Exception as e:
                logger.warning(f"⚠️ 加载归一化参数失败，重新计算：{str(e)}")

        # 计算归一化参数（使用前N个样本）
        logger.info(f"📊 计算归一化参数（策略：{'全局' if self.use_global_normalization else '子带'}，分量：{self.norm_components}，样本数：{self.norm_sample_count}）")
        sample_count = min(self.norm_sample_count, len(self.data_indices))
        norm_dataset = []

        for idx in self.data_indices[:sample_count]:
            # 加载所有归一化分量的子带数据
            comp_subbands = []
            valid = True
            for comp in self.norm_components:
                sb = self._load_single_component(comp, idx)
                if sb is None:
                    valid = False
                    break
                comp_subbands.append(sb)
            if valid:
                norm_dataset.append(comp_subbands)

        if not norm_dataset:
            raise RuntimeError(f"无足够样本计算归一化参数（需至少1个有效样本，当前：{len(norm_dataset)}）")

        # 计算参数（子带级/全局级）
        if self.use_global_normalization:
            norm_params = compute_subband_norm_params(
                norm_dataset,
                self.norm_components
            )
        else:
            norm_params = compute_subband_norm_params(
                norm_dataset,
                self.norm_components
            )

        # 保存参数
        if self.save_norm_params:
            os.makedirs(os.path.dirname(self.norm_params_path), exist_ok=True)
            with open(self.norm_params_path, 'wb') as f:
                pickle.dump(norm_params, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"✅ 归一化参数已保存至：{self.norm_params_path}（覆盖分量：{list(norm_params.keys())}）")

        return norm_params

    def __len__(self) -> int:
        return len(self.data_indices)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], int, Dict[str, Any]]:
        """
        获取单个样本
        Returns:
            input_sbs: 输入子带列表（每个元素：(2, H, W)，vx+vz）
            target_sbs: 目标子带列表（每个元素：(2, H, W)，curl+div）
            data_idx: 样本索引
            meta: 元数据
        """
        max_retries = 3
        retry_count = 0
        current_idx = idx

        while retry_count < max_retries:
            try:
                data_idx = self.data_indices[current_idx]
                meta = {"sample_idx": data_idx, "subband_count": utils.SUBBAND_COUNT}

                # 加载所有分量的子带数据
                comp_subbands = {}
                for comp in self.all_components:
                    sb = self._load_single_component(comp, data_idx)
                    if sb is None:
                        raise RuntimeError(f"分量{comp}加载失败")
                    comp_subbands[comp] = sb

                # 应用变换（仅翻转，无噪声）
                if self.transform is not None:
                    transform_input = [comp_subbands[comp] for comp in self.all_components]
                    transform_output = self.transform(transform_input)
                    for i, comp in enumerate(self.all_components):
                        comp_subbands[comp] = transform_output[i]

                # 1. 构建输入子带（vx + vz双通道，归一化后合并）
                input_sbs = []
                for sb_idx in range(utils.SUBBAND_COUNT):
                    # 归一化vx和vz子带（高频增强+归一化）
                    vx_sb = normalize_subband(
                        comp_subbands["vx"][sb_idx],
                        self.norm_params,
                        sb_idx,
                        "vx",
                        self.high_freq_enhance_factor,
                        self.use_global_normalization
                    )
                    vz_sb = normalize_subband(
                        comp_subbands["vz"][sb_idx],
                        self.norm_params,
                        sb_idx,
                        "vz",
                        self.high_freq_enhance_factor,
                        self.use_global_normalization
                    )
                    # 合并为2通道：(2, H, W)
                    combined_sb = np.stack([vx_sb, vz_sb], axis=0)
                    input_sbs.append(torch.from_numpy(combined_sb))

                # 2. 构建目标子带（curl + div双通道，归一化后合并）
                target_sbs = []
                for sb_idx in range(utils.SUBBAND_COUNT):
                    # 归一化curl和div子带（高频增强+归一化）
                    curl_sb = normalize_subband(
                        comp_subbands["curl"][sb_idx],
                        self.norm_params,
                        sb_idx,
                        "curl",
                        self.high_freq_enhance_factor,
                        self.use_global_normalization
                    )
                    div_sb = normalize_subband(
                        comp_subbands["div"][sb_idx],
                        self.norm_params,
                        sb_idx,
                        "div",
                        self.high_freq_enhance_factor,
                        self.use_global_normalization
                    )
                    # 合并为2通道：(2, H, W)（与模型输出维度匹配）
                    combined_target = np.stack([curl_sb, div_sb], axis=0)
                    target_sbs.append(torch.from_numpy(combined_target))

                return input_sbs, target_sbs, data_idx, meta

            except Exception as e:
                # 记录失败的原样本索引，重试下一个样本
                failed_idx = self.data_indices[current_idx]
                retry_count += 1
                current_idx = (current_idx + 1) % len(self.data_indices)
                logger.warning(f"⚠️ 加载样本{failed_idx}失败，重试{retry_count}/{max_retries}：{str(e)}")

        raise RuntimeError(f"连续{max_retries}个样本加载失败，无法获取有效样本")

# ==============================================
# 导出类和变量（供其他模块导入）
# ==============================================
__all__ = [
    "CurveletAwareTransform",
    "SeismicCurveletDataloader",
]

# ==============================================
# 自测代码：运行本文件即可执行全部测试
# ==============================================
if __name__ == "__main__":
    import tempfile
    import shutil

    # ======================
    # 测试用子带配置（512×64对应的25个子带）
    # ======================
    TEST_SUBBAND_SIZES = [
        (21, 171), (43, 151), (43, 151), (19, 343), (19, 343),
        (43, 151), (43, 151), (19, 343), (19, 343), (43, 279),
        (43, 279), (43, 279), (43, 279), (35, 343), (35, 343),
        (35, 343), (35, 343), (43, 279), (43, 279), (43, 279),
        (43, 279), (35, 343), (35, 343), (35, 343), (35, 343)
    ]
    TEST_SCALE_COUNTS = [1, 8, 16]
    TEST_SAMPLE_COUNT = 8

    # 测试结果统计
    test_results = {}

    def run_test(test_name: str, test_func):
        """执行单个测试并记录结果"""
        print(f"\n{'='*60}")
        print(f"🧪 开始测试：{test_name}")
        try:
            test_func()
            print(f"✅ 测试通过：{test_name}")
            test_results[test_name] = True
        except Exception as e:
            print(f"❌ 测试失败：{test_name}")
            print(f"   错误原因：{str(e)}")
            import traceback
            traceback.print_exc()
            test_results[test_name] = False

    # ======================
    # 辅助函数：生成模拟测试数据
    # ======================
    def generate_mock_data(root_dir: str):
        components = ["vx", "vz", "curl", "div"]
        for comp in components:
            comp_dir = os.path.join(root_dir, comp)
            os.makedirs(comp_dir, exist_ok=True)
            for idx in range(TEST_SAMPLE_COUNT):
                subbands = []
                for h, w in TEST_SUBBAND_SIZES:
                    # 生成带幅值的随机数据，模拟真实曲波子带
                    sb = np.random.randn(h, w).astype(np.float32) * (0.5 / (idx + 1))
                    subbands.append(sb)
                # 保存为object数组（与真实数据格式一致）
                np.save(os.path.join(comp_dir, f"{comp}_{idx}.npy"), np.array(subbands, dtype=object))
        print(f"   生成模拟数据：{TEST_SAMPLE_COUNT}个样本，4个分量，25个子带")

    # ======================
    # 测试1：数据集基础初始化
    # ======================
    def test_basic_init():
        tmp_dir = tempfile.mkdtemp()
        try:
            generate_mock_data(tmp_dir)
            dataset = SeismicCurveletDataloader(
                data_path=tmp_dir,
                subband_sizes=TEST_SUBBAND_SIZES,
                scale_dir_counts=TEST_SCALE_COUNTS,
                norm_sample_count=3
            )
            assert len(dataset) == TEST_SAMPLE_COUNT, f"样本数错误：预期{TEST_SAMPLE_COUNT}，实际{len(dataset)}"
            assert utils.SUBBAND_COUNT == 25, f"子带总数错误：预期25，实际{utils.SUBBAND_COUNT}"
        finally:
            shutil.rmtree(tmp_dir)

    # ======================
    # 测试2：子带数量与尺寸校验
    # ======================
    def test_subband_size_check():
        tmp_dir = tempfile.mkdtemp()
        try:
            generate_mock_data(tmp_dir)
            dataset = SeismicCurveletDataloader(
                data_path=tmp_dir,
                subband_sizes=TEST_SUBBAND_SIZES,
                scale_dir_counts=TEST_SCALE_COUNTS,
                norm_sample_count=3
            )
            input_sbs, target_sbs, _, _ = dataset[0]
            assert len(input_sbs) == 25, f"输入子带数量错误：预期25，实际{len(input_sbs)}"
            assert len(target_sbs) == 25, f"目标子带数量错误：预期25，实际{len(target_sbs)}"
            # 校验每个子带尺寸
            for sb_idx in range(25):
                expected_h, expected_w = TEST_SUBBAND_SIZES[sb_idx]
                assert input_sbs[sb_idx].shape[1:] == (expected_h, expected_w), \
                    f"子带{sb_idx}尺寸错误：预期({expected_h},{expected_w})，实际{input_sbs[sb_idx].shape[1:]}"
        finally:
            shutil.rmtree(tmp_dir)

    # ======================
    # 测试3：输入输出通道校验
    # ======================
    def test_channel_check():
        tmp_dir = tempfile.mkdtemp()
        try:
            generate_mock_data(tmp_dir)
            dataset = SeismicCurveletDataloader(
                data_path=tmp_dir,
                subband_sizes=TEST_SUBBAND_SIZES,
                scale_dir_counts=TEST_SCALE_COUNTS,
                norm_sample_count=3
            )
            input_sbs, target_sbs, _, _ = dataset[0]
            # 输入应为2通道（vx + vz）
            assert input_sbs[0].shape[0] == 2, f"输入通道数错误：预期2，实际{input_sbs[0].shape[0]}"
            # 目标应为2通道（curl + div）
            assert target_sbs[0].shape[0] == 2, f"目标通道数错误：预期2，实际{target_sbs[0].shape[0]}"
            # 数据类型校验
            assert input_sbs[0].dtype == torch.float32, f"输入数据类型错误：预期float32，实际{input_sbs[0].dtype}"
        finally:
            shutil.rmtree(tmp_dir)

    # ======================
    # 测试5：数据变换（翻转）有效性
    # ======================
    def test_transform():
        tmp_dir = tempfile.mkdtemp()
        try:
            generate_mock_data(tmp_dir)
            # 无变换
            dataset_no_trans = SeismicCurveletDataloader(
                data_path=tmp_dir,
                subband_sizes=TEST_SUBBAND_SIZES,
                scale_dir_counts=TEST_SCALE_COUNTS,
                transform=None,
                norm_sample_count=3
            )
            input1, _, _, _ = dataset_no_trans[0]
            sb0_original = input1[0].numpy().copy()

            # 强制100%翻转
            transform = CurveletAwareTransform(flip_prob=1.0)
            dataset_trans = SeismicCurveletDataloader(
                data_path=tmp_dir,
                subband_sizes=TEST_SUBBAND_SIZES,
                scale_dir_counts=TEST_SCALE_COUNTS,
                transform=transform,
                norm_sample_count=3
            )
            input2, _, _, _ = dataset_trans[0]
            sb0_flipped = input2[0].numpy().copy()

            # 验证翻转生效
            assert not np.allclose(sb0_original, sb0_flipped), "翻转未生效"
            assert np.allclose(sb0_original[:, :, ::-1], sb0_flipped), "翻转结果不正确"
        finally:
            shutil.rmtree(tmp_dir)

    # ======================
    # 测试6：批量加载稳定性
    # ======================
    def test_batch_load():
        tmp_dir = tempfile.mkdtemp()
        try:
            generate_mock_data(tmp_dir)
            dataset = SeismicCurveletDataloader(
                data_path=tmp_dir,
                subband_sizes=TEST_SUBBAND_SIZES,
                scale_dir_counts=TEST_SCALE_COUNTS,
                max_data=5,
                norm_sample_count=3
            )
            assert len(dataset) == 5, f"max_data限制失效：预期5，实际{len(dataset)}"
            # 遍历全部样本
            for i in range(len(dataset)):
                input_sbs, target_sbs, idx, meta = dataset[i]
                assert len(input_sbs) == 25
                assert len(target_sbs) == 25
                assert meta["subband_count"] == 25
        finally:
            shutil.rmtree(tmp_dir)

    # ======================
    # 测试7：外部初始化配置兼容性
    # ======================
    def test_external_init():
        tmp_dir = tempfile.mkdtemp()
        try:
            generate_mock_data(tmp_dir)
            # 外部先初始化
            init_subband_config(TEST_SUBBAND_SIZES, TEST_SCALE_COUNTS)
            # 数据集不传尺寸参数
            dataset = SeismicCurveletDataloader(
                data_path=tmp_dir,
                norm_sample_count=3
            )
            assert len(dataset) == TEST_SAMPLE_COUNT
            assert utils.SUBBAND_COUNT == 25
        finally:
            shutil.rmtree(tmp_dir)

    # ======================
    # 执行全部测试
    # ======================
    print("🚀 开始执行数据集全面自测")
    run_test("基础初始化与样本数", test_basic_init)
    run_test("子带数量与尺寸校验", test_subband_size_check)
    run_test("输入输出通道与数据类型", test_channel_check)
    run_test("数据变换（翻转）有效性", test_transform)
    run_test("批量加载稳定性", test_batch_load)
    run_test("外部配置初始化兼容性", test_external_init)

    # ======================
    # 测试总结
    # ======================
    print(f"\n{'='*60}")
    print("📋 测试结果汇总")
    passed = sum(test_results.values())
    total = len(test_results)
    for name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status}  {name}")
    print(f"\n总计：{passed}/{total} 项测试通过")
    if passed == total:
        print("🎉 全部测试通过，数据集代码适配正常！")
    else:
        print(f"⚠️  有 {total - passed} 项测试失败，请检查对应逻辑")