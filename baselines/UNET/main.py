import os
import torch
import datetime
import numpy as np
from sklearn.metrics import r2_score
from torchsummary import summary
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
torch.set_default_dtype(torch.float32)

# 本地模块
from paras import parameters
from cusdataloader import *
from utils import *   # 包含 init_net, set_requires_grad, lambda_rule 等

from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5'

# ==================== UNet 定义（新增） ====================
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
    """上采样：双线性插值 + 跳跃连接 + DoubleConv"""
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
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
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
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        p_out = self.outc_p(x)
        s_out = self.outc_s(x)
        return p_out, s_out
# ============================================================

def local_para(opt=None):
    """
    参数配置函数，用于设置模型训练和测试所需的各类超参数及路径。
    """
    ## data parameter ##
    opt.train_model_id = 432
    opt.test_model_id = 432
    opt.out_name = 'NNmodel'
    opt.prepredix = f'trainOn_m{opt.train_model_id}'
    
    # 数据路径（已指向预处理后的 512×64 空间域数据）
    opt.data_path = f'./data/train_model_{opt.train_model_id}_preprocessed_512x64'
    opt.data_path_valid = f'./data/test_model_{opt.test_model_id}_preprocessed_512x64'
    
    # 动态获取训练数据集大小
    train_vz_path = os.path.join(opt.data_path, "vz")
    if os.path.exists(train_vz_path):
        opt.max_data = len([f for f in os.listdir(train_vz_path) 
                           if f.startswith('vz_') and f.endswith('.npy')])
    else:
        opt.max_data = 0
    
    # 动态获取验证数据集大小
    valid_vz_path = os.path.join(opt.data_path_valid, "vz")
    if os.path.exists(valid_vz_path):
        opt.max_data_valid = len([f for f in os.listdir(valid_vz_path) 
                                 if f.startswith('vz_') and f.endswith('.npy')])
    else:
        opt.max_data_valid = 0
        print(f"警告: 验证路径 {valid_vz_path} 不存在")
    
    # 模型训练设置
    opt.if_continue = False
    opt.continue_net_path_G = "old_model.pth"
    opt.epoches = 5000
    opt.batch_size = 16
    opt.gpuid = 2
    opt.nph = 2
    opt.nop = 1
    opt.nt = 512
    opt.nr = 64
    opt.num_workers = 4
    opt.num_gpu = 1

    opt.patience = 200
    opt.early_stop_counter = 0
    opt.early_stop = False

    opt.data_start = 0
    opt.jsample = 1
    opt.lr = 0.001
    opt.beta1 = 0.5

    opt.do_validate = 1
    opt.data_start_valid = 0
    opt.jsample_valid = 10
    opt.seed = 0
    opt.validate_interval = 5
    opt.sample_interval = 10

    return opt


def main():
    """
    主训练函数，实现完整的神经网络训练流程
    """
    opt = parameters().get_parser()
    opt = local_para(opt)
    opt.out_predix = opt.prepredix

    print("------ All read-in parameters ------\n", opt.__dict__)
    torch.cuda.set_device(opt.gpuid)
    mydevice = torch.device('cuda:' + str(opt.gpuid) if torch.cuda.is_available() else "cpu")
    print(f"------ using device: {mydevice} ------")

    # 数据加载
    if opt.nph == 2:
        myDataset = npy_2to2_Dataloader_getindex(opt.data_path, opt.nt, opt.nr, opt.nph, opt.nop,
                                                 opt.max_data, opt.data_norm_type, opt.data_start, opt.jsample, 0)
    elif opt.nph == 1:
        myDataset = npy_1to2_Dataloader_getindex(opt.data_path, opt.nt, opt.nr, opt.nph, opt.nop,
                                                 opt.max_data, opt.data_norm_type, opt.data_start, opt.jsample, 0)
    else:
        print("ERROR: opt.nph should be 1 or 2.")
    
    mySampler = RandomSampler(myDataset)
    mydataloader = DataLoader(myDataset, batch_size=int(opt.batch_size),
                               shuffle=None, sampler=mySampler, batch_sampler=None,
                               num_workers=int(opt.num_workers), pin_memory=False,
                               drop_last=False, timeout=0, worker_init_fn=None)
    
    if opt.nph == 2:
        myDataset_valid = npy_2to2_Dataloader_getindex(
            opt.data_path_valid, opt.nt, opt.nr, opt.nph, opt.nop,
            opt.max_data_valid, opt.data_norm_type,
            opt.data_start_valid, opt.jsample_valid, 0
        )
    elif opt.nph == 1:
        myDataset_valid = npy_1to2_Dataloader_getindex(opt.data_path_valid, opt.nt, opt.nr, opt.nph, opt.nop,
                                                 opt.max_data_valid, opt.data_norm_type, opt.data_start_valid, opt.jsample_valid, 0)
    else:
        print("ERROR: opt.nph should be 1 or 2.")

    mySampler_valid = SequentialSampler(myDataset_valid)
    mydataloader_valid = DataLoader(
        myDataset_valid,
        batch_size=int(opt.batch_size),
        sampler=mySampler_valid,
        num_workers=min(4, int(opt.num_workers)),
        pin_memory=False,
        drop_last=False
    )

    save_dir = os.path.join(os.getcwd(), opt.out_predix)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))

    # ========== 使用标准 UNet 替代原 FCN ==========
    net_G = UNet(opt.nph, opt.nop)
    init_net(net_G, gpu_ids=[opt.gpuid])
    print(f"Network: {net_G.__class__.__name__}")
    total_params = sum(p.numel() for p in net_G.parameters())
    print(f"参数量: {total_params/1e6:.2f} M")  # 应约为 31.04M

    set_requires_grad(net_G, requires_grad=True)

    # 训练配置
    metric_gan = torch.nn.MSELoss().to(mydevice)
    optimizer = torch.optim.Adam(net_G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer.zero_grad()
    scheduler_G = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    print("------ training begin ------")
    start_time = datetime.datetime.now()

    # 早停参数
    patience = 10
    early_stop_counter = 0
    best_r2 = -float('inf')
    best_epoch = -1
    best_val_loss = float('inf')
    early_stop = False

    for epoch in range(opt.epoch_start if hasattr(opt, 'epoch_start') else 0, opt.epoches):
        epoch_loss = 0.0
        batch_i = 1

        for data_B, data_A1, data_A2, fname in mydataloader:
            data_B, data_A1, data_A2 = data_B.to(mydevice), data_A1.to(mydevice), data_A2.to(mydevice)

            fake_A1, fake_A2 = net_G(data_B)

            loss_G1 = metric_gan(fake_A1, data_A1)
            loss_G2 = metric_gan(fake_A2, data_A2)
            loss = loss_G1 + loss_G2
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            r2_p = r2_score(data_A1.cpu().detach().numpy().flatten(), fake_A1.cpu().detach().numpy().flatten())
            r2_s = r2_score(data_A2.cpu().detach().numpy().flatten(), fake_A2.cpu().detach().numpy().flatten())
            if batch_i % 10 == 0:
                writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(mydataloader) + batch_i)
                writer.add_scalar('R2_P/train_batch', r2_p, epoch * len(mydataloader) + batch_i)
                writer.add_scalar('R2_S/train_batch', r2_s, epoch * len(mydataloader) + batch_i)

            elapsed_time = datetime.datetime.now() - start_time
            print("Rank %d: [Epoch %d/%d] [Batch %d/%d] [G loss: %f, %f] [r2 score: p %f, s %f]time: % s"
                  % (opt.gpuid, epoch, opt.epoches, batch_i, int(opt.max_data / opt.batch_size),
                     loss_G1, loss_G2, r2_p, r2_s, elapsed_time))
            batch_i += 1

        avg_epoch_loss = epoch_loss / len(mydataloader)
        writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)

        # 验证环节
        if opt.do_validate and epoch % opt.validate_interval == 0:
            r2lp = []; r2ls = []; loss_p = []; loss_s = []

            for data_B, data_A1, data_A2, fname in mydataloader_valid:
                data_B, data_A1, data_A2 = data_B.to(mydevice), data_A1.to(mydevice), data_A2.to(mydevice)

                set_requires_grad(net_G, False)
                fake_A1, fake_A2 = net_G(data_B)

                loss_p_val = metric_gan(fake_A1, data_A1).item()
                loss_s_val = metric_gan(fake_A2, data_A2).item()
                loss_p.append(loss_p_val)
                loss_s.append(loss_s_val)

                r2lp.append(r2_score(data_A1.cpu().detach().numpy().flatten(),
                                fake_A1.cpu().detach().numpy().flatten()))
                r2ls.append(r2_score(data_A2.cpu().detach().numpy().flatten(),
                                fake_A2.cpu().detach().numpy().flatten()))

            avg_r2_p = np.mean(r2lp)
            avg_r2_s = np.mean(r2ls)
            current_r2 = (avg_r2_p + avg_r2_s) / 2
            current_val_loss = np.mean(loss_p) + np.mean(loss_s)

            if current_r2 > best_r2:
                best_r2 = current_r2
                best_epoch = epoch
                best_val_loss = current_val_loss
                early_stop_counter = 0

                torch.save(net_G.state_dict(), os.path.join(save_dir, "best_model.pth"))
                with open(os.path.join(save_dir, "best_info.txt"), "w") as f:
                    f.write(f"Best epoch: {epoch}\n")
                    f.write(f"Best R²: {best_r2:.6f}\n")
                    f.write(f"Validation loss: {best_val_loss:.6f}\n")
                print(f"✅ 更新最优模型 (Epoch {epoch}, R²={best_r2:.6f})")
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stopping triggered after {patience} consecutive validation checks without improvement.")
                    early_stop = True

            writer.add_scalar('Loss/val_p', np.mean(loss_p), epoch)
            writer.add_scalar('Loss/val_s', np.mean(loss_s), epoch)
            writer.add_scalar('R2_P/val', avg_r2_p, epoch)
            writer.add_scalar('R2_S/val', avg_r2_s, epoch)

            print("Validation: Rank %d: [Epoch %d/%d] [loss P: %f] [loss S: %f]"
                  % (opt.gpuid, epoch, opt.epoches, np.mean(loss_p), np.mean(loss_s)))
            print("Validation: Rank %d: [Epoch %d/%d] The average of [r2p score: %f] [r2s score:% f]"
                  % (opt.gpuid, epoch, opt.epoches, avg_r2_p, avg_r2_s))
            print("Validation: Rank %d: [Epoch %d/%d] The median of [r2p score: %f] [r2s score: %f]"
                  % (opt.gpuid, epoch, opt.epoches, np.median(r2lp), np.median(r2ls)))
            print("Validation: Rank %d: [Epoch %d/%d] The maximum of: [r2p score: %f at %d] [r2s score: % f at % d]"
                  % (opt.gpuid, epoch, opt.epoches, np.max(r2lp), np.argmax(r2lp), np.max(r2ls), np.argmax(r2ls)))
            print("Validation: Rank %d: [Epoch %d/%d] The minimum of: [r2p score: %f at %d] [r2s score: % f at % d]"
                  % (opt.gpuid, epoch, opt.epoches, np.min(r2lp), np.argmin(r2lp), np.min(r2ls), np.argmin(r2ls)))

            set_requires_grad(net_G, True)

            if early_stop:
                break

        if early_stop:
            break

        scheduler_G.step()
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

    writer.close()
    print("------ training finished ------")
    print(f"🏆 最佳模型信息：Epoch {best_epoch}, R² = {best_r2:.6f}, 验证损失 = {best_val_loss:.6f}")
    print(f"📂 最佳模型保存于：{os.path.join(save_dir, 'best_model.pth')}")
    print(f"📄 详细信息记录于：{os.path.join(save_dir, 'best_info.txt')}")


if __name__ == '__main__':
    main()