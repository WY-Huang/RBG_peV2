import math
import os
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import colour_demosaicing as cds


class Stress2Fringe(nn.Module):
    """
    进行应力图到条纹图的计算
    """
    def __init__(self, device):
        super(Stress2Fringe, self).__init__()
        self.device = device

    def stress2fringe(self, Stress_map, interaction):
        """
        :param interaction: Calculating the normalized spectral interaction between light source and camera sensor
        :param Stress_map: Continuous surface or gray map.
        """
        # 0> 设置参数
        Stress_Max = 72e6           # Maximun stress value that could exist within the experiments (in Pa).
        Stress_Magnitude = 72e6     # Stress magnitude to scale the continuous surface (in Pa). ？？？没看懂
        h = 0.01                    # Body thickness in m.
        C = 3.5e-12                 # Stress optic coefficient (in m^2/N).

        # 1> 处理网络输出的re_stressmap
        Stress_map = Stress_map.squeeze()

        # 2> Preprocessing input parameters
        rows, columns = 224, 224
        stress_in = (Stress_map - Stress_map.min()) / (Stress_map.max() - Stress_map.min())     # 归一化到[0, 1]
        stress_in = stress_in * Stress_Magnitude

        Spectral_range = torch.linspace(390, 760, 371)      # 光谱的波长

        # 3> Generating the isochromatic image from the (dark field polariscope)  暗场应该是：无背景光强Ia
        Isochromatic = torch.zeros(rows, columns, 3, device=self.device)
        for i in range(371):
            phase = 2 * math.pi * h * C * stress_in / (1e-9 * Spectral_range[i])
            for j in range(3):
                Isochromatic[:, :, j] = Isochromatic[:, :, j] + (interaction[i, j] / 2) * (1 - torch.cos(phase))

        isochromatic = Isochromatic / Isochromatic.max()

        stressMap = stress_in / Stress_Max

        # # 4> Introducing the Bayer effetc using a 'grbg' cfa filter
        # Img_bayer = torch.zeros(rows, columns)
        # Img_bayer[::2, ::2] = isochromatic[::2, ::2, 1]
        # Img_bayer[1::2, 1::2] = isochromatic[1::2, 1::2, 1]
        # Img_bayer[::2, 1::2] = isochromatic[::2, 1::2, 0]
        # Img_bayer[1::2, ::2] = isochromatic[1::2, ::2, 2]
        #
        # # 5> Introducing the demosaicking effetc from a 'grbg' cfa filter.
        # iso_fringe = cds.demosaicing_CFA_Bayer_Malvar2004(Img_bayer, 'GRBG')
        # # iso_fringe = (iso_fringe - iso_fringe.min()) / (iso_fringe.max() - iso_fringe.min())
        # iso_fringe = np.minimum(iso_fringe, 1)      # 大于1的数取1
        # iso_fringe = np.maximum(iso_fringe, 0)      # 小于0的数取0

        # 后处理
        # iso_fringe = torch.from_numpy(iso_fringe)
        iso_fringe = isochromatic.permute(2, 0, 1).unsqueeze(0)

        return iso_fringe, stressMap

    def forward(self, stressmap, ss_interaction):

        fringes, _ = self.stress2fringe(stressmap, ss_interaction)

        return fringes


def load_SSdata():
    """
    加载光源和传感器数据, 并计算光源和传感器之间的归一化光谱作用
    Source: Data vector with the relative spectral content sampled into 371 cells (371x1).
    Sensor: Data array with the relative spectral response of camera sensor sampled into 371 cells
                       per color component(371x3).
    """
    # 光源数据(可能由于精确度不够，导致问题)
    source_data = pd.read_csv("pe_data/processed_source_data.csv")
    Incandescent_source = torch.tensor(source_data['Incandescent'])
    # Fluorescent_source = torch.tensor(source_data['Fluorescent'])
    # WhiteLED_source = torch.tensor(source_data['WhiteLED'])

    source = Incandescent_source

    # 传感器数据(可能由于精确度不够，导致问题)
    sensor_data = pd.read_csv("pe_data/DCC3260C_seneor_data.csv")
    sensor = torch.zeros(371, 3)
    sensor[:, 0] = torch.tensor(sensor_data['R_channel'])
    sensor[:, 1] = torch.tensor(sensor_data['G_channel'])
    sensor[:, 2] = torch.tensor(sensor_data['B_channel'])

    Source = source / source.max()
    S_r = sensor[:, 0]
    S_g = sensor[:, 1]
    S_b = sensor[:, 2]

    # 2> Calculating the normalized spectral interaction between light source and camera sensor
    Interaction = torch.zeros(371, 3)
    Interaction[:, 0] = Source * S_r
    Interaction[:, 1] = Source * S_g
    Interaction[:, 2] = Source * S_b
    interaction = Interaction / Interaction.max()

    return interaction


def copy_codes(path):
    """
    将当前目录下的所有的py文件复制到指定目录
    """
    s_dir = os.getcwd()
    t_dir = f"{path}/codes"
    os.makedirs(t_dir, exist_ok=True)
    for file in os.listdir(s_dir):
        if os.path.splitext(file)[1] == ".py":
            shutil.copy(file, t_dir)
