import scipy.io
import numpy as np
import cv2
import os
from PIL import Image
from torch.utils.data import Dataset
from src.data.components import Transform

def read_data_file(root_dir):
    arr = os.listdir(os.path.join(root_dir, 'images'))
    arr.sort()
    arr = ["./data/lsp/images/" + dir for dir in arr]
    return arr

def read_mat_file(root_dir, img_list):
    mat = scipy.io.loadmat(os.path.join(root_dir, 'joints.mat'))['joints']
    kpts = mat.transpose([2, 0, 1])
    
    centers =[]
    for idx in range(kpts.shape[0]):
        im = Image.open(img_list[idx])
        w, h = im.size[0], im.size[1]
        
        center_x = (kpts[idx][kpts[idx][:, 0] < w, 0].max() +
                    kpts[idx][kpts[idx][:, 0] > 0, 0].min()) / 2
        center_y = (kpts[idx][kpts[idx][:, 1] < h, 1].max() +
                    kpts[idx][kpts[idx][:, 1] > 0, 1].min()) / 2
        centers.append([center_x, center_y])

    return kpts, centers

def gaussian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)

class LSP_Data(Dataset):
    def __init__(self, root_dir, stride, sigma):
        self.img_list = read_data_file(root_dir)
        self.kpt_list, self.center_list = read_mat_file(root_dir, self.img_list)
        self.stride = stride
        self.sigma = sigma
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        kpt = self.kpt_list[idx]
        center = self.center_list[idx]
        img = np.array(cv2.imread(img_path), dtype=np.float32)
        
        transform = Transform.Resized(368)
        img, kpt, center = transform(img, kpt, center)

        height, width, _ = img.shape
        
        heatmap = np.zeros((int(height/self.stride), int(width/self.stride), int(len(kpt)+1)), dtype=np.float32)
        for i in range(len(kpt)):
            # resize from 368 to 46
            x = int(kpt[i][0]) * 1.0 / self.stride
            y = int(kpt[i][1]) * 1.0 / self.stride
            heat_map = gaussian_kernel(size_h = int(height/self.stride), size_w = int(width/self.stride), 
                                       center_x = x, center_y = y, sigma = self.sigma)
            heat_map[heat_map > 1] = 1
            heat_map[heat_map < 0.0099] = 0
            heatmap[:, :, i + 1] = heat_map

        heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)  # for background

        img = Transform.normalize(Transform.to_tensor(img), 
                                  [128.0, 128.0, 128.0],
                                  [256.0, 256.0, 256.0])
        
        heatmap = Transform.to_tensor(heatmap)

        return img, heatmap

    def __len__(self):
        return len(self.img_list)