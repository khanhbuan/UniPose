import scipy.io
import os
from PIL import Image

def read_data_file(root_dir):
    arr = os.listdir(os.path.join(root_dir, 'images'))
    arr.sort()
    return arr

def read_mat_file(root_dir, img_list):
    mat = scipy.io.loadmat(os.path.join(root_dir, 'joints.mat'))['joints']
    kpts = mat.transpose([2, 0, 1])
    
    centers =[]
    for idx in range(kpts.shape[0]):
        im = Image.open(os.path.join(root_dir, 'images', img_list[idx]))
        w, h = im.size[0], im.size[1]
        
        center_x = (kpts[idx][kpts[idx][:, 0] < w, 0].max() +
                    kpts[idx][kpts[idx][:, 0] > 0, 0].min()) / 2
        center_y = (kpts[idx][kpts[idx][:, 1] < h, 1].max() +
                    kpts[idx][kpts[idx][:, 1] > 0, 1].min()) / 2
        centers.append([center_x, center_y])

    return kpts, centers



# if __name__ == "__main__":
    