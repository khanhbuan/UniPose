import scipy.io
import os

def read_data_file(root_dir):
    arr = os.listdir(os.path.join(root_dir, 'images'))
    arr.sort()
    return arr

def read_mat_file(root_dir):
    mat = scipy.io.loadmat(os.path.join(root_dir, 'joints.mat'))['joints']
    print(mat.shape)

if __name__ == "__main__":
    arr = read_mat_file(root_dir="./data/lsp")
    print(arr)