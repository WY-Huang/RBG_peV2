# =========================================================================
"""
6762张彩色条纹图到应力图的程序，光源为 Incandescent_source，相机传感器为 DCC3260C
"""
# =========================================================================
import time
import os

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
from ms_ssim import MS_SSIM

# tensorboard --logdir=result/0907-2145
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # torch.set_default_dtype(torch.float32)
    master_path = "result/" + "%s" % time.strftime("%m%d-%H%M")

    # 保存当前运行代码
    copy_codes(master_path)

    # 1> 读取数据(1, c, h, w)
    batch_size = 32
    datapath = "pe_data/data_2000"
    trans = transforms.ToTensor()
    trainsets = PeDataSet(datapath, trans)
    train_dataloader = DataLoader(trainsets, batch_size, shuffle=True, num_workers=4, drop_last=True)

    #  1> 光源和传感器数据
    ss_interaction = load_SSdata()
    ss_interaction = ss_interaction.to(device)
    stress2fringe = Stress2Fringe(device).to(device)

    # 2> 定义网络模型
    net = ResStressnet(3, 1)  # res18unet
    net = net.to(device)

    # 3> 定义损失函数和优化器
    inital_lr = 0.000001
    epochs = 150

    # criterion = torch.nn.MSELoss()
    criterion = MS_SSIM()
    criterion = criterion.to(device)
    optimizer = optim.Adam(net.parameters(), lr=inital_lr)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4> 训练
    loss_init = 1000
    log_dir = master_path + "/logs/"
    writer = SummaryWriter(log_dir)

    save_path = master_path + "/models"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 训练循环
    for epoch in range(epochs):
        print(f"start epoch {epoch}")
        # curr_lr = scheduler.get_last_lr()[0]        # 获取当前学习率

        Loss = []
        Loss_mean = 0
        train_bar = tqdm(train_dataloader)
        for step, data in enumerate(train_bar):
            fringe_img, truth_stressmap = data
            fringe_img = fringe_img.to(device)
            truth_stressmap = truth_stressmap.to(device)

            optimizer.zero_grad()
            re_stressmap = net(fringe_img)

            # 带入光学模型计算
            # time_start = time.time()
            # re_fringe = torch.zeros(batch_size, 3, 224, 224, device=device)
            # for b in range(batch_size):
            #     re_fringe[b] = stress2fringe(re_stressmap[b], ss_interaction)       # 循环调用，每张图需要0.25s

            re_fringe = stress2fringe(re_stressmap, ss_interaction)

            # time_end = time.time()
            # print("stressmap2fringe take: ", time_end - time_start, "s")

            re_loss = criterion(re_fringe, fringe_img)
            re_loss.backward()
            optimizer.step()

            # tensorboard写入信息
            detach_loss = re_loss.item()
            Loss.append(detach_loss)
            Loss_mean = np.mean(Loss)

            # 输出tqdm的信息
            train_bar.set_postfix_str(
                f"Epoch: {epoch}, Lr: {inital_lr}, Loss_mean: {Loss_mean}, "
                f"Step: {step}, Loss_train: {detach_loss:.5f}")

            writer.add_scalar(f'Loss_by/step', np.array(detach_loss), len(train_dataloader) * epoch + step)
            # writer.add_scalar(f'Loss_perEpoch/loss_epoch{epoch}', np.array(detach_loss), step)

            # 拼接输入条纹图，重建应力图，重建条纹图，真实应力图
            if epoch > 0.7 * epochs and step < 2:
                three_vs = torch.cat((fringe_img, re_fringe), dim=0)
                two_vs = torch.cat((truth_stressmap, re_stressmap), dim=0)
                img_grid = utils.make_grid(three_vs, nrow=batch_size)
                img_grid2 = utils.make_grid(two_vs, nrow=batch_size)
                writer.add_image(f"{epoch}_Fringes/RawFringe+ReFringe_{step}", img_grid)
                writer.add_image(f"{epoch}_StressMaps/truth_stressmap+Restressmap_{step}", img_grid2)

        if Loss_mean < loss_init and epoch > 0.7 * epochs:
            loss_init = Loss_mean
            tqdm.write(f"-----save model epoch_{epoch}-----")
            torch.save(net.state_dict(), save_path + "/loss_best_model.pth")
        # tensorboard写入信息
        # writer.add_scalar('Lr', np.array(curr_lr), epoch)
        # scheduler.step()

        writer.add_scalar('Loss_by/epoch', np.mean(Loss), epoch)
