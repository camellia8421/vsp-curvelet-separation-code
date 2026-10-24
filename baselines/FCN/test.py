import os
import torch
import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, RandomSampler

torch.set_default_dtype(torch.float32)

from cusdataloader import *
from paras import parameters
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5'


def local_para(opt=None):
    ## 本地参数设置 ##
    opt.which_epoch = 1
    opt.train_model_id = 62
    opt.test_model_id = 432
    opt.continue_net_path_G = './trainOn_m62/best_model.pth'
    opt.out_name = 'Test'
    opt.prepredix = f'Results_trainOn_m{opt.train_model_id}_'
    opt.data_path = './data/test_model_432_preprocessed_512x64'
    opt.max_data = 141
    opt.batch_size = 1
    opt.gpuid = 2
    opt.nph = 2
    opt.nop = 1
    opt.nt = 512
    opt.nr = 64
    opt.padded_nr = 64
    opt.num_workers = 4
    opt.num_gpu = 1
    opt.data_start = 0
    opt.jsample = 1
    opt.save_testout = 0
    opt.seed = 0
    opt.validate_interval = 1
    opt.sample_interval = 1
    return opt


def extract_sample_id(fname_raw):
    """
    从 batchfname 中提取数字 ID，用于排序与CSV记录
    """
    fname_str = str(fname_raw)
    try:
        return int(fname_str)
    except ValueError:
        pass
    import re
    nums = re.findall(r'\d+', fname_str)
    if nums:
        return int(nums[0])
    else:
        return -1


def main():
    opt = parameters().get_parser()
    opt = local_para(opt)
    opt.out_predix = opt.prepredix + 'testOn_m' + str(opt.test_model_id) + '_epoch_' + str(opt.which_epoch)
    print("------ All read-in parameters ------\n", opt.__dict__)

    torch.cuda.set_device(opt.gpuid)
    mydevice = torch.device('cuda:' + str(opt.gpuid) if torch.cuda.is_available() else "cpu")
    print(f"------ using device: {mydevice} ------")

    if opt.nph == 2:
        myDataset = npy_2to2_Dataloader_getindex(opt.data_path, opt.nt, opt.nr, opt.nph, opt.nop,
                                                 opt.max_data, opt.data_norm_type, opt.data_start, opt.jsample, 0)
    elif opt.nph == 1:
        myDataset = npy_1to2_Dataloader_getindex(opt.data_path, opt.nt, opt.nr, opt.nph, opt.nop,
                                                 opt.max_data, opt.data_norm_type, opt.data_start, opt.jsample, 0)
    else:
        print("ERROR: opt.nph should be 1 or 2.")
        return

    mySampler = RandomSampler(myDataset)
    mydataloader = DataLoader(myDataset, batch_size=int(opt.batch_size),
                              shuffle=None, sampler=mySampler, batch_sampler=None,
                              num_workers=int(opt.num_workers), pin_memory=False,
                              drop_last=False, timeout=0, worker_init_fn=None)

    net_G = FCN8s_conv_upsample_out2_v2_k5_c5(opt.nph, opt.nop)
    init_net(net_G, gpu_ids=[opt.gpuid])
    print(f"Network: {net_G.__class__.__name__}")
    set_requires_grad(net_G, False)

    modelG_loadin = torch.load(opt.continue_net_path_G, map_location=lambda storage, loc: storage.cuda(opt.gpuid))
    net_G.load_state_dict(modelG_loadin)

    save_dir = os.path.join(os.getcwd(), opt.out_predix)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    print("------ test begin ------")
    start_time = datetime.datetime.now()
    results = []  # 存储指标，每个元素包含 sample_id 和指标

    for w, (batchdata_B, batchdata_A1, batchdata_A2, batchfname) in enumerate(mydataloader):
        batchdata_B, batchdata_A1, batchdata_A2 = batchdata_B.to(mydevice), batchdata_A1.to(mydevice), batchdata_A2.to(mydevice)
        net_G.eval()
        with torch.no_grad():
            batchfake_A1, batchfake_A2 = net_G(batchdata_B)

        for this_subbatch in range(opt.batch_size):
            fname_raw = batchfname[this_subbatch].numpy()
            sample_id = extract_sample_id(fname_raw)
            fname = str(fname_raw)

            # 取出单样本，去掉通道维度（batch_size=1, nop=1）
            data_A1 = batchdata_A1[this_subbatch].squeeze()   # shape: (512, 64)
            fake_A1 = batchfake_A1[this_subbatch].squeeze()
            data_A2 = batchdata_A2[this_subbatch].squeeze()
            fake_A2 = batchfake_A2[this_subbatch].squeeze()

            # 使用完整 512×64 数据计算指标
            data_A1_np = data_A1.cpu().detach().numpy()
            fake_A1_np = fake_A1.cpu().detach().numpy()
            data_A2_np = data_A2.cpu().detach().numpy()
            fake_A2_np = fake_A2.cpu().detach().numpy()

            # 计算指标
            r2_p = r2_score(data_A1_np.flatten(), fake_A1_np.flatten())
            r2_s = r2_score(data_A2_np.flatten(), fake_A2_np.flatten())

            mse_p = np.mean((data_A1_np - fake_A1_np) ** 2)
            mse_s = np.mean((data_A2_np - fake_A2_np) ** 2)

            max_p = np.max(data_A1_np)
            max_s = np.max(data_A2_np)
            psnr_p = 20 * np.log10(max_p / (np.sqrt(mse_p) + 1e-8))
            psnr_s = 20 * np.log10(max_s / (np.sqrt(mse_s) + 1e-8))

            signal_power_p = np.mean(data_A1_np ** 2)
            signal_power_s = np.mean(data_A2_np ** 2)
            snr_p = 10 * np.log10(signal_power_p / (mse_p + 1e-8))
            snr_s = 10 * np.log10(signal_power_s / (mse_s + 1e-8))

            results.append({
                'sample_id': sample_id,
                'filename': fname,
                'r2p': r2_p,
                'r2s': r2_s,
                'psnr_p': psnr_p,
                'psnr_s': psnr_s,
                'snr_p': snr_p,
                'snr_s': snr_s
            })

            # 保存预测和标签（完整尺寸）
            if opt.save_testout:
                np.save(os.path.join(save_dir, f"{fname}_P_pred.npy"), fake_A1_np)
                np.save(os.path.join(save_dir, f"{fname}_S_pred.npy"), fake_A2_np)
                np.save(os.path.join(save_dir, f"{fname}_P_label.npy"), data_A1_np)
                np.save(os.path.join(save_dir, f"{fname}_S_label.npy"), data_A2_np)

    # 创建DataFrame并按 sample_id 排序
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('sample_id', inplace=True)

    # 保存全部样本的CSV
    csv_path = os.path.join(save_dir, 'metrics_all.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ 全部样本指标已保存至: {csv_path}")

    # 打印全部样本的统计
    print("\n------ 整体统计（所有样本） ------")
    print(f"样本总数: {len(df)}")
    for col in ['r2p', 'r2s', 'psnr_p', 'psnr_s', 'snr_p', 'snr_s']:
        print(f"{col}: 平均={df[col].mean():.6f}, 中位数={df[col].median():.6f}, 最大={df[col].max():.6f}, 最小={df[col].min():.6f}")

    print("------ finished ------")
    print(f"耗时: {datetime.datetime.now() - start_time}")


if __name__ == '__main__':
    main()
