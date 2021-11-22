import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Res18_U_net import ResStressnet
from PeDataSet import PeDataSet
from pe_utils import Stress2Fringe, load_SSdata

# 1> 加载训练好的模型
# device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
model_path = "result/1120-1348/models/loss_best_model.pth"

net = ResStressnet(3, 1)
# net = net.to(device)
net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

# 2> 加载测试数据
test_path = "pe_data/data_test"
test_data = PeDataSet(test_path, transforms.ToTensor())
test_dataloader = DataLoader(test_data, 1)

# 3> 读取传感器和光源数据
ss_interaction = load_SSdata()

# 4> 进行预测
with torch.no_grad():
    for data in test_dataloader:
        # data = data.to(device)
        fringe, stressmap = data

        # 预测
        net.eval()
        predict_stressmap = net(fringe)

        fringe_out = Stress2Fringe("cpu")(predict_stressmap, ss_interaction)

        # 绘图
        plt.figure(figsize=(12, 3))
        plt.subplot(141)
        plt.title("raw_fringe")
        plt.imshow(fringe.squeeze().permute(1, 2, 0))

        plt.subplot(142)
        plt.title("raw_stressmap")
        plt.imshow(stressmap.squeeze(0).permute(1, 2, 0), "gray")

        plt.subplot(143)
        plt.title("predict_stressmap")
        plt.imshow(predict_stressmap.squeeze(0).permute(1, 2, 0), "gray")

        plt.subplot(144)
        plt.title("recover_fringe")
        plt.imshow(fringe_out.squeeze(0).permute(1, 2, 0))

        plt.show()
