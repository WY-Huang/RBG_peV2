############################################################################
"""
6762张彩色条纹图到应力图的程序，光源为 Incandescent_source，相机传感器为 DCC3260C
"""
############################################################################
import time

import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils, transforms

from PeDataSet import PeDataSet
from Res18_U_net import ResStressnet
from pe_utils import Stress2Fringe, copy_codes, load_SSdata

# tensorboard --logdir=result/0907-2145
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    master_path = "result/" + "%s" % time.strftime("%m%d-%H%M")

    # 1> 读取数据(1, c, h, w)
    batch_size = 16
    datapath = "pe_data/data_100"
    trans = transforms.ToTensor()
    trainsets = PeDataSet(datapath, trans)
    train_dataloader = DataLoader(trainsets, batch_size, shuffle=True, num_workers=4, drop_last=True)

    #    光源和传感器数据
    ss_interaction = load_SSdata()
    ss_interaction = ss_interaction.to(device)
    stress2fringe = Stress2Fringe(device).to(device)

    # 2> 定义网络模型
    net = ResStressnet(3, 1)  # res18unet
    net = net.to(device)

    # 3> 定义损失函数和优化器
    criterion = torch.nn.MSELoss()
    # criterion = MS_SSIM()
    # criterion = criterion.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    # 4> 训练
    epochs = 1
    Loss = []
    curr_lr = 0

    log_dir = master_path + "/logs/"
    writer = SummaryWriter(log_dir)

    for epoch in range(epochs):
        print(f"start epoch {epoch}")
        train_bar = tqdm(train_dataloader)
        for step, data in enumerate(train_bar):
            fringe_img, truth_stressmap = data
            fringe_img = fringe_img.to(device)
            truth_stressmap = truth_stressmap.to(device)

            optimizer.zero_grad()
            re_stressmap = net(fringe_img)

            # 带入光学模型计算
            time_start = time.time()
            re_fringe = torch.zeros(batch_size, 3, 224, 224, device=device)
            for b in range(batch_size):
                re_fringe[b] = stress2fringe(re_stressmap[b], ss_interaction)       # 循环调用，每张图需要0.25s

            time_end = time.time()
            # print("stressmap2fringe take: ", time_end - time_start, "s")

            re_loss = criterion(re_fringe, fringe_img)
            re_loss.backward()
            optimizer.step()
            scheduler.step()

            # tensorboard写入信息
            detach_loss = re_loss.item()
            Loss.append(detach_loss)
            writer.add_scalar(f'LOSS/loss_{epoch}', np.array(detach_loss), step)

            curr_lr = scheduler.get_last_lr()[0]
            writer.add_scalar(f'LR/lr_{epoch}', np.array(curr_lr), step)

            # 拼接输入条纹图，重建应力图，重建条纹图，真实应力图
            # if epoch % 10 == 0:
            three_vs = torch.cat((fringe_img, re_fringe), dim=0)

            two_vs = torch.cat((re_stressmap, truth_stressmap), dim=0)
            img_grid = utils.make_grid(three_vs, nrow=batch_size)
            img_grid2 = utils.make_grid(two_vs, nrow=batch_size)
            writer.add_image(f"{epoch}_Fringes/RawFringe+ReFringe_{step}", img_grid)
            writer.add_image(f"{epoch}_StressMaps/Restressmap+truth_stressmap_{step}", img_grid2)

            train_bar.set_postfix_str(
                f"Epoch: {epoch}, Step: {step}, Lr: {curr_lr}, Loss_train: {detach_loss:.5f}")

        writer.add_scalar('LR', np.array(curr_lr), epoch)
        writer.add_scalar('LOSS', np.mean(Loss), epoch)

    # 保存当前运行代码
    copy_codes(master_path)
