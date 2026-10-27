import torch
import torch.nn.functional as F
# from kmeans_gpu import kmeans
import math
import numpy as np
import time
import sys
import os
import logging
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def NormalizeFeaTorch(features):
    rowsum = torch.tensor((features ** 2).sum(1))
    r_inv = torch.pow(rowsum, -0.5)
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    normalized_feas = torch.mm(r_mat_inv, features)
    return normalized_feas



def get_Similarity(fea_mat1, fea_mat2):
    Sim_mat = F.cosine_similarity(fea_mat1.unsqueeze(1), fea_mat2.unsqueeze(0), dim=-1)
    return Sim_mat



def euclidean_dist(x, y, root=False):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    if root:
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist



def next_batch1(X1, X2, batch_size):
    """Return data for next batch"""
    tot = X1.shape[0]

    total = math.ceil(tot / batch_size)
    X2 = torch.tensor(X2)
    X2 = X2.t()
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)

        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]

        yield (batch_x1, X2, (i + 1))
    # tot = X1.shape[0]
    # total = math.ceil(tot / batch_size)
    # for i in range(int(total)):
    #     start_idx = i * batch_size
    #     end_idx = (i + 1) * batch_size
    #     end_idx = min(tot, end_idx)
    #     batch_x1 = X1[start_idx: end_idx, ...]
    #     batch_x2 = X2[start_idx: end_idx, ...]
    #     yield (batch_x1, batch_x2, (i + 1))


def next_batch(X1, X2, Proto, batch_size, Knowledge_size):
    """Return data for next batch"""
    tot = X1.shape[0]
    tot_p = Proto.shape[0]
    total = math.ceil(tot / batch_size)
    X2 = torch.tensor(X2)
    X2 = X2.t()
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)

        start_idx_p = i * Knowledge_size
        end_idx_p = (i + 1) * Knowledge_size
        end_idx_p = min(end_idx_p, tot_p)

        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        # batch_Atten = Attention[start_idx: end_idx, ...]
        batch_Proto = Proto[start_idx_p: end_idx_p, ...]
        # Handle remaining data

        yield (batch_x1, X2, batch_Proto, (i + 1))

def cal_std(acc, nmi, ari):
    """Return the mean and standard deviation of ACC, NMI and ARI."""
    return tuple(value for scores in (acc, nmi, ari)
                 for value in (np.mean(scores), np.std(scores)))

def get_logger(config):

    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")
        if not os.path.exists('./logs/' + str(config.datasets[1]) + '/'):
            os.mkdir('./logs/' + str(config.datasets[1]) + '/')
    path = os.path.join("./logs/" + str(config.datasets[1]) + '/')
    if not os.path.exists(path):
        os.makedirs(path)

    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    current_time = current_time.replace(":", "_")
    filename = config.datasets[1] + "_time=" + current_time + ".csv"
    results_df = pd.DataFrame(columns=["View", "ACC", "NMI", "ARI"])
    fh = logging.FileHandler(os.path.join(path, filename))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

def Make_file(config):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_root = os.path.join(base_dir, "logs")
    if not os.path.exists(log_root):
        os.mkdir(log_root)
    path = os.path.join(log_root, str(config.datasets[1]))
    if not os.path.exists(path):
        os.makedirs(path)

    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()

    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    current_time = current_time.replace(":", "_")
    alpha = f"{config.similarity_alpha:.1f}".replace(".", "_")
    filename = config.datasets[1] + "_alpha=" + alpha + "_time=" + current_time + ".csv"
    results_df = pd.DataFrame(
        columns=["Alpha", "ACC", "NMI", "ARI"])
    fh = logging.FileHandler(os.path.join(path, filename))
    fh.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(fh)

    path = path + os.sep
    results_df.to_csv(path + filename, index=False)
    logging.info(config)

    return path, filename



def tensor_to_numpy(tensor_data):
    """将PyTorch张量转换为numpy数组，自动处理GPU和梯度"""
    if hasattr(tensor_data, 'is_cuda') and tensor_data.is_cuda:
        # GPU张量：移动到CPU，分离梯度，转换为numpy
        return tensor_data.cpu().detach().numpy()
    elif hasattr(tensor_data, 'detach'):
        # CPU张量：分离梯度，转换为numpy
        return tensor_data.detach().numpy()
    elif hasattr(tensor_data, 'numpy'):
        # 已经是numpy兼容的类型
        return tensor_data.numpy()
    else:
        # 直接返回（可能是已经是numpy数组）
        return tensor_data
