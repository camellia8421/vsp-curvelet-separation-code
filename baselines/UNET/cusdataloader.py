import torch  
import numpy as np 

class npy_2to2_Dataloader_getindex():
    """
    自定义数据加载器类，用于加载双分量输入地震数据（如vz和vx分量）
    并将它们转换为适合神经网络输入的张量格式[6](@ref)
    
    属性:
        data_path_B1: vz分量数据路径
        data_path_B2: vx分量数据路径
        data_path_A1: div分量数据路径
        data_path_A2: curl分量数据路径
        nt: 时间步数
        nr: 接收器数量
        nph: 输入通道数（固定为2）
        nop: 输出通道数（固定为1）
        max_data: 最大数据量
        norm_type: 数据归一化类型
        data_start: 数据起始索引
        jsample: 采样间隔
        seek_num: 寻址偏移量（默认为0）
    """
    
    def __init__(self, data_path, nt, nr, nph, nop, max_data, data_norm_type, data_start, jsample, seek_num=0):
        """初始化数据加载器实例[6](@ref)"""
        self.data_path_B1 = data_path + "/vz"  # 垂直分量数据路径
        self.data_path_B2 = data_path + "/vx"  # 水平分量数据路径
        self.data_path_A1 = data_path + "/div"  # 散度分量数据路径（通常对应P波）
        self.data_path_A2 = data_path + "/curl"  # 旋度分量数据路径（通常对应S波）
        self.nt = nt  # 时间步数
        self.nr = nr  # 接收器数量
        self.nph = nph  ## 2  # 输入通道数（固定为2，代表双分量输入）
        self.nop = nop  ## 1  # 输出通道数（固定为1）
        self.max_data = max_data  # 最大数据量
        self.norm_type = data_norm_type  # 数据归一化类型
        self.data_start = data_start  # 数据起始索引
        self.jsample = jsample  # 采样间隔
        self.seek_num = seek_num  # 寻址偏移量（默认为0）

    def __len__(self):
        """返回实际可采样的样本数（修正后）"""
        if self.max_data <= self.data_start:
            return 0  # 起始索引超出文件总数，无有效样本
        # 计算最大有效索引（不超过max_data-1）
        max_valid_index = self.max_data - 1
        # 计算可采样的样本数：(最大有效索引 - 起始索引) // 采样间隔 + 1
        sample_count = ((max_valid_index - self.data_start) // self.jsample) + 1
        return sample_count if sample_count > 0 else 0

    def get2exp(self, num):
        """
        找到大于等于num的最小2的幂次数，用于内存对齐优化[4](@ref)
        
        参数:
            num: 输入数值
            
        返回:
            大于等于num的最小2的幂次数，最大不超过2^11+2^10
        """
        for exp in range(0, 12):  # 遍历指数从0到11
            if num <= 2 ** exp:  # 如果num小于等于2的exp次幂
                return 2 ** exp  # 返回2的exp次幂
                break  # 跳出循环
            if num > 2 ** 11:  # 如果num大于2的11次幂（2048）
                return 2 ** 11 + 2 ** 10  # 返回2048+1024=3072

    def __getitem__(self, index):
        # 1. 计算实际文件索引
        this_index = self.data_start + index * self.jsample
        # 2. 关键校验：确保this_index不超出实际文件范围（0 ~ max_data-1）
        if this_index >= self.max_data:
            raise IndexError(
                f"采样索引超出范围！this_index={this_index}，"
                f"实际最大有效索引={self.max_data-1}，"
                f"请检查data_start={self.data_start}、jsample={self.jsample}、max_data={self.max_data}"
            )
        if this_index < 0:
            raise IndexError(f"this_index={this_index} 不能为负数，请检查data_start配置")
        """
        获取指定索引的数据样本[6](@ref)
        
        参数:
            index: 样本索引
            
        返回:
            B_tensor: 输入特征张量（双分量地震数据）
            A1_tensor: 目标输出1张量（div分量，通常对应P波）
            A2_tensor: 目标输出2张量（curl分量，通常对应S波）
            this_index: 实际使用的数据索引
        """
        this_index = self.data_start + index * self.jsample  # 计算实际数据索引
        # 初始化输入张量（双通道）
        B_tensor = torch.zeros((self.nph, self.get2exp(self.nt), self.get2exp(self.nr)))
        # 初始化输出张量1（单通道）
        A1_tensor = torch.zeros((1, self.get2exp(self.nt), self.get2exp(self.nr)))
        # 初始化输出张量2（单通道）
        A2_tensor = torch.zeros((1, self.get2exp(self.nt), self.get2exp(self.nr)))
        
        # 从npy文件加载数据
        B1 = np.load(self.data_path_B1 + "/vz_" + str(this_index) + ".npy")  # 加载垂直分量数据
        B2 = np.load(self.data_path_B2 + "/vx_" + str(this_index) + ".npy")  # 加载水平分量数据
        A1 = np.load(self.data_path_A1 + "/div_" + str(this_index) + ".npy")  # 加载散度分量数据
        A2 = np.load(self.data_path_A2 + "/curl_" + str(this_index) + ".npy")  # 加载旋度分量数据

        # 计算填充起始位置（使数据在张量中居中）
        start_a = int((self.get2exp(self.nt) - self.nt) / 2)  # 时间维度填充起始位置
        start_b = int((self.get2exp(self.nr) - self.nr) / 2)  # 接收器维度填充起始位置
        
        # 将数据放入张量中（只填充有效区域，周围为零填充）
        B_tensor[0, start_a: start_a + self.nt, start_b: start_b + self.nr] = torch.from_numpy(B1)  # 垂直分量
        B_tensor[1, start_a: start_a + self.nt, start_b: start_b + self.nr] = torch.from_numpy(B2)  # 水平分量
        A1_tensor[:, start_a: start_a + self.nt, start_b: start_b + self.nr] = torch.from_numpy(A1)  # 散度分量
        A2_tensor[:, start_a: start_a + self.nt, start_b: start_b + self.nr] = torch.from_numpy(A2)  # 旋度分量
        
        return B_tensor, A1_tensor, A2_tensor, this_index  # 返回数据张量和索引


class npy_1to2_Dataloader_getindex():
    """
    自定义数据加载器类，用于加载单分量输入地震数据（如仅vz分量）
    并将它们转换为适合神经网络输入的张量格式[6](@ref)
    
    属性:
        data_path_B1: vz分量数据路径
        data_path_A1: div分量数据路径
        data_path_A2: curl分量数据路径
        nt: 时间步数
        nr: 接收器数量
        nph: 输入通道数（固定为1）
        nop: 输出通道数（固定为1）
        max_data: 最大数据量
        norm_type: 数据归一化类型
        data_start: 数据起始索引
        jsample: 采样间隔
        seek_num: 寻址偏移量（默认为0）
    """
    
    def __init__(self, data_path, nt, nr, nph, nop, max_data, data_norm_type, data_start, jsample, seek_num=0):
        """初始化数据加载器实例[6](@ref)"""
        self.data_path_B1 = data_path + "/vz"  # 垂直分量数据路径
        # self.data_path_B2 = data_path + "/vx"  # 水平分量数据路径（注释掉，单分量不使用）
        self.data_path_A1 = data_path + "/div"  # 散度分量数据路径（通常对应P波）
        self.data_path_A2 = data_path + "/curl"  # 旋度分量数据路径（通常对应S波）
        self.nt = nt  # 时间步数
        self.nr = nr  # 接收器数量
        self.nph = 1 #nph  ## constant  # 输入通道数（固定为1，代表单分量输入）
        self.nop = 1 #nop  ## constant  # 输出通道数（固定为1）
        self.max_data = max_data  # 最大数据量
        self.norm_type = data_norm_type  # 数据归一化类型
        self.data_start = data_start  # 数据起始索引
        self.jsample = jsample  # 采样间隔
        self.seek_num = seek_num  # 寻址偏移量（默认为0）

    def __len__(self):
        """返回实际可采样的样本数（修正后）"""
        if self.max_data <= self.data_start:
            return 0  # 起始索引超出文件总数，无有效样本
        # 计算最大有效索引（不超过max_data-1）
        max_valid_index = self.max_data - 1
        # 计算可采样的样本数：(最大有效索引 - 起始索引) // 采样间隔 + 1
        sample_count = ((max_valid_index - self.data_start) // self.jsample) + 1
        return sample_count if sample_count > 0 else 0

    def get2exp(self, num):
        """
        找到大于等于num的最小2的幂次数，用于内存对齐优化[4](@ref)
        
        参数:
            num: 输入数值
            
        返回:
            大于等于num的最小2的幂次数，最大不超过2^11+2^10
        """
        for exp in range(0, 12):  # 遍历指数从0到11
            if num <= 2 ** exp:  # 如果num小于等于2的exp次幂
                return 2 ** exp  # 返回2的exp次幂
                break  # 跳出循环
            if num > 2 ** 11:  # 如果num大于2的11次幂（2048）
                return 2 ** 11 + 2 ** 10  # 返回2048+1024=3072

    def __getitem__(self, index):
        # 1. 计算实际文件索引
        this_index = self.data_start + index * self.jsample
        # 2. 关键校验：确保this_index不超出实际文件范围（0 ~ max_data-1）
        if this_index >= self.max_data:
            raise IndexError(
                f"采样索引超出范围！this_index={this_index}，"
                f"实际最大有效索引={self.max_data-1}，"
                f"请检查data_start={self.data_start}、jsample={self.jsample}、max_data={self.max_data}"
            )
        if this_index < 0:
            raise IndexError(f"this_index={this_index} 不能为负数，请检查data_start配置")
        """
        获取指定索引的数据样本[6](@ref)
        
        参数:
            index: 样本索引
            
        返回:
            B_tensor: 输入特征张量（单分量地震数据）
            A1_tensor: 目标输出1张量（div分量，通常对应P波）
            A2_tensor: 目标输出2张量（curl分量，通常对应S波）
            this_index: 实际使用的数据索引
        """
        this_index = self.data_start + index * self.jsample  # 计算实际数据索引
        # 初始化输入张量（单通道）
        B_tensor = torch.zeros((self.nph, self.get2exp(self.nt), self.get2exp(self.nr)))
        # 初始化输出张量1（单通道）
        A1_tensor = torch.zeros((1, self.get2exp(self.nt), self.get2exp(self.nr)))
        # 初始化输出张量2（单通道）
        A2_tensor = torch.zeros((1, self.get2exp(self.nt), self.get2exp(self.nr)))
        
        # 从npy文件加载数据
        B1 = np.load(self.data_path_B1 + "/vz_" + str(this_index) + ".npy")  # 加载垂直分量数据
        # B2 = np.load(self.data_path_B2 + "/vx_" + str(this_index) + ".npy")  # 水平分量数据（注释掉，单分量不使用）
        A1 = np.load(self.data_path_A1 + "/div_" + str(this_index) + ".npy")  # 加载散度分量数据
        A2 = np.load(self.data_path_A2 + "/curl_" + str(this_index) + ".npy")  # 加载旋度分量数据

        # 计算填充起始位置（使数据在张量中居中）
        start_a = int((self.get2exp(self.nt) - self.nt) / 2)  # 时间维度填充起始位置
        start_b = int((self.get2exp(self.nr) - self.nr) / 2)  # 接收器维度填充起始位置
        
        # 将数据放入张量中（只填充有效区域，周围为零填充）
        B_tensor[:, start_a: start_a + self.nt, start_b: start_b + self.nr] = torch.from_numpy(B1)  # 垂直分量
        # B_tensor[1, start_a: start_a + self.nt, start_b: start_b + self.nr] = torch.from_numpy(B2)  # 水平分量（注释掉）
        A1_tensor[:, start_a: start_a + self.nt, start_b: start_b + self.nr] = torch.from_numpy(A1)  # 散度分量
        A2_tensor[:, start_a: start_a + self.nt, start_b: start_b + self.nr] = torch.from_numpy(A2)  # 旋度分量
        
        return B_tensor, A1_tensor, A2_tensor, this_index  # 返回数据张量和索引