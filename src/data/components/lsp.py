import os
import scipy.io
from PIL import Image
from torch.utils.data import Dataset

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

class LSP_Data(Dataset):
    def __init__(self, root_dir):
        self.img_list = read_data_file(root_dir)
        self.kpt_list, self.center_list = read_mat_file(root_dir, self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        kpt = self.kpt_list[idx]
        center = self.center_list[idx]

        return img_path, kpt, center

    def __len__(self):
        return len(self.img_list)