import cv2
import torch
import matplotlib.pyplot as plt

from pe_utils import Stress2Fringe, load_SSdata


# 1> 读取应力图和条纹图
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

stressmap = cv2.imread("pe_data/data_100/stressmaps/Target_500.bmp", cv2.IMREAD_GRAYSCALE)
fringe = cv2.imread("pe_data/data_100/fringes/Img_500.bmp")
fringe = cv2.cvtColor(fringe, cv2.COLOR_BGR2RGB)

# 2> 读取传感器和光源数据
ss_interaction = load_SSdata()

# 3> 进行条纹图生成计算
stress_ipt = torch.from_numpy(stressmap)
stress_ipt = stress_ipt.unsqueeze(0)

fringe_out = Stress2Fringe(device)(stress_ipt, ss_interaction)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(fringe_out.squeeze(0).permute(2, 1, 0))

plt.subplot(1, 3, 2)
plt.imshow(fringe)

plt.subplot(1, 3, 3)
plt.imshow(stressmap, "gray")

plt.show()