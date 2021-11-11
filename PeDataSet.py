import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PeDataSet(Dataset):
    """
    加载光弹性数据集
    """
    def __init__(self, rootdir, transform=None, getname=None):
        self.rootdir = rootdir
        self.fringes = os.listdir(rootdir + '/fringes')
        self.transform = transform
        self.getname = getname

    def __len__(self):
        return len(self.fringes)

    def __getitem__(self, index):
        fringeindex = self.fringes[index]       # 根据索引获取图片路径
        imgnames = fringeindex.split('.')[0]
        img = Image.open(self.rootdir + '/fringes/' + fringeindex)
        img = self.transform(img)

        stressmapIndex = 'Target_' + fringeindex.split('_')[1]
        stressmap = Image.open(self.rootdir + '/stressmaps/' + stressmapIndex)
        stressmap = self.transform(stressmap)

        if self.getname:
            return img, imgnames
        else:
            return img, stressmap


if __name__ == "__main__":
    trans = transforms.ToTensor()
    datasets = PeDataSet("pe_data", transform=trans)
    img10, stress = datasets[100]

    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.imshow(img10.permute(1, 2, 0))

    plt.subplot(122)
    plt.imshow(stress.permute(1, 2, 0), cmap="gray")
    # plt.title(f"{imgname}")
    plt.show()
