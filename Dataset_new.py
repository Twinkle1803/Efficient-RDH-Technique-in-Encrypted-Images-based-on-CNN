from torch.utils.data import Dataset
import os
import cv2
import numpy as np
### 自定义数据集
class MyData(Dataset):
    ## 类中定义一些变量
    def __init__(self, root_dir, transforms_=None):
        self.transform = transforms_
        self.root_dir = root_dir
        self.img_path = os.listdir(self.root_dir)   ## 列出所有图像

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, img_name) ## 单张图像的具体地址
        img_self = cv2.imread(img_item_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_item_path, cv2.IMREAD_GRAYSCALE)
        img = np.array(img, dtype=np.float32)
        img = np.expand_dims(img, axis=2)
        img = self.transform(img)
        return img, img_self,img_name
    def __len__(self):
        return len(self.img_path)

