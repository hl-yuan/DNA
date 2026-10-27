import scipy.io as sio
import torch
import warnings
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler


warnings.filterwarnings("ignore")


# path = './datasets/'

ALL_data = dict(
    YoutubeFace_sel_fea={1: 'YoutubeFace_sel_fea', 'N': 101499, 'K': 31, 'V': 5, 'n_input': [64, 512, 64, 647, 838], "Knowledge_size": 31, "beta1": 1e-0, "beta2": 1e-0, "similarity_alpha": 0.5},
    NUS={1: 'NUS', 'N': 2400, 'K': 12, 'V': 6, 'n_input': [64, 144, 73, 128, 225, 500], "Knowledge_size": 12, "beta1": 1e-0, "beta2": 1e-3, "similarity_alpha": 0.3},
    OutdoorScene={1: 'OutdoorScene', 'N': 2688, 'K': 8, 'V': 4, 'n_input': [512, 432, 256, 48], "Knowledge_size": 8, "beta1": 1e-0, "beta2": 1e-3, "similarity_alpha": 0.0},
    ALOI_10={1: 'ALOI_10', 'N': 1079, 'K': 10, 'V': 4, 'n_input': [64, 64, 77, 13], "Knowledge_size": 10, "beta1": 1e-0, "beta2": 1e-1, "similarity_alpha": 0.6},
    ALOI_100={1: 'ALOI_100', 'N': 10800, 'K': 100, 'V': 4, 'n_input': [77, 13, 64, 125], "Knowledge_size": 100, "beta1": 1e-0, "beta2": 1e-0, "similarity_alpha": 0.6},
    oxford4={1: 'oxford4', 'N': 800, 'K': 4, 'V': 3, 'n_input': [1764, 10, 128], "Knowledge_size": 4, "beta1": 1e-0, "beta2": 1e-0, "similarity_alpha": 0.9},
)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_DATASETS = os.path.join(_BASE_DIR, 'Datasets')
_SOURCE_DATASETS = os.path.abspath(os.path.join(_BASE_DIR, '..', 'DNA_Train_main', 'Datasets'))
path = _LOCAL_DATASETS if os.path.exists(_LOCAL_DATASETS) else _SOURCE_DATASETS


def load_data2(dataset):
    if dataset[1] not in ALL_data:
        raise ValueError(f"Unsupported dataset: {dataset[1]}")
    if dataset[1] == 'YoutubeFace_sel_fea':
        return load_datav4(dataset)
    return load_datav1(dataset)


def load_datav1(dataset):
    file_path = os.path.join(path, dataset[1])
    mat_data = sio.loadmat(file_path)
    X_raw = mat_data['X']
    Y_raw = mat_data['Y']
    Label = np.array(Y_raw).reshape(-1)

    X = []
    mm = MinMaxScaler()
    # nn = mat_data['X'].shape[1]

    for i in range(dataset['V']):
        from scipy.sparse import csc_matrix
        if isinstance(X_raw[0, i], csc_matrix):
            view_data = np.array(X_raw[0, i].toarray(), dtype=np.float32)
        else:
            view_data = np.array(X_raw[0, i], dtype=np.float32)

        view_data = mm.fit_transform(view_data)  # 归一化
        X.append(view_data)
    Y = [Label for _ in range(len(X))]
    return X, Y

def load_datav4(dataset):
    file_path = os.path.join(path, dataset[1])
    mat_data = sio.loadmat(file_path)
    X_raw = mat_data['X']
    Y_raw = mat_data['Y']
    Label = np.array(Y_raw).reshape(-1)

    X = []
    mm = MinMaxScaler()
    # nn = mat_data['X'].shape[1]

    for i in range(dataset['V']):
        view_data = np.array(X_raw[i, 0], dtype=np.float32).T
        print(view_data.shape)
        view_data = mm.fit_transform(view_data)  # 归一化
        X.append(view_data)
    Y = [Label for _ in range(len(X))]
    return X, Y


class TrainDataset_Com(torch.utils.data.Dataset):
    def __init__(self, X_list, Y_list):
        self.X_list = X_list
        self.Y_list = Y_list
        # self.Z_list = Z_list

    def __getitem__(self, index):
        current_x_list = []
        current_y_list = []
        current_z_list = []
        current_x = self.X_list[index]
        current_x_list.append(current_x)
        current_y = self.Y_list
        current_y_list.append(current_y)
        # current_z = self.Z_list
        # current_z_list.append(current_z)

        # X_list1 = self.X_list
        # Y_list1 = self.Y_list
        return current_x_list, current_y_list

    def __len__(self):
        # return the total size of data
        return self.X_list.shape[0]

class Data_Sampler(object):

    def __init__(self, pairs, shuffle=False, batch_size=1, drop_last=False):
        if shuffle:
            self.sampler = RandomSampler(pairs)
        else:
            self.sampler = SequentialSampler(pairs)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batch = [batch]
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            batch = [batch]
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

