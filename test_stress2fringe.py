import cv2
import torch
import matplotlib.pyplot as plt

from pe_utils import Stress2Fringe, load_SSdata


# 1> 读取应力图和条纹图
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

stressmap = cv2.imread("pe_data/data_100/stressmaps/Target_500.bmp", cv2.IMREAD_GRAYSCALE)
fringe = cv2.imread("pe_data/data_100/fringes/Img_500.bmp")
fringe = cv2.cvtColor(fringe, cv2.COLOR_BGR2RGB)

# 2> 读取传感器和光源数据
ss_interaction = load_SSdata()

# 3> 进行条纹图生成计算
# torch.from_numpy(array)是做数组的浅拷贝，torch.Tensor(array)是做数组的深拷贝
# stress_ipt = torch.from_numpy(stressmap)        # stressmap和stress_ipt是共用内存的，修改会相互改变
stress_ipt = torch.tensor(stressmap, dtype=torch.float32)
stress_ipt = stress_ipt.unsqueeze(0).unsqueeze(0)

fringe_out, stress_out = Stress2Fringe(device)(stress_ipt, ss_interaction)

plt.figure(figsize=(16, 4))
plt.subplot(1, 4, 1)
plt.title("recover_fringe")
plt.imshow(fringe_out.squeeze(0).permute(1, 2, 0))

plt.subplot(1, 4, 2)
plt.title("predict_stressmap")
plt.imshow(stress_out.squeeze(0), "gray")

plt.subplot(1, 4, 3)
plt.title("raw_fringe")
plt.imshow(fringe)

plt.subplot(1, 4, 4)
plt.title("raw_stressmap")
plt.imshow(stressmap, "gray")

plt.show()
