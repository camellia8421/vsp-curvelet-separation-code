import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# ==================== UNet 模型定义 ====================

class DoubleConv(nn.Module):
    """(卷积 -> BN -> ReLU) × 2"""
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """下采样：最大池化 + DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    """上采样：双线性插值（或转置卷积）+ 跳跃连接 + DoubleConv"""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理尺寸不一致（若有）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """输出卷积（1×1）"""
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """标准UNet，输出两个分支（P波和S波）"""
    def __init__(self, nph, nop, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = nph
        self.n_classes = nop
        self.bilinear = bilinear

        # 编码器
        self.inc = DoubleConv(nph, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)   # 若用双线性，底部为512

        # 解码器
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # 两个输出分支
        self.outc_p = OutConv(64, nop)
        self.outc_s = OutConv(64, nop)

    def forward(self, x):
        # 编码
        x1 = self.inc(x)          # 64
        x2 = self.down1(x1)       # 128
        x3 = self.down2(x2)       # 256
        x4 = self.down3(x3)       # 512
        x5 = self.down4(x4)       # 512

        # 解码
        x = self.up1(x5, x4)      # 256
        x = self.up2(x, x3)       # 128
        x = self.up3(x, x2)       # 64
        x = self.up4(x, x1)       # 64

        # 两个输出
        p_out = self.outc_p(x)
        s_out = self.outc_s(x)
        return p_out, s_out


# ==================== 辅助函数（用于训练） ====================

def set_requires_grad(nets, requires_grad=False):
    """
    设置网络中所有参数的 requires_grad 属性
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def init_weights(net, init_type='normal', gain=0.02):
    """
    初始化网络权重
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    初始化网络并移动到指定 GPU
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
    init_weights(net, init_type, gain=init_gain)
    return net


def lambda_rule(epoch):
    """
    学习率衰减规则（用于 LambdaLR）
    """
    lr_l = 1.0 - max(0, epoch + 1 - 1000) / float(1000 + 1)
    return lr_l