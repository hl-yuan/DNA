import math

import torch
import torch.nn.functional as F
from utils import get_Similarity
import torch.nn as nn



class Loss_Reconstruction(nn.Module):

    def __init__(self):
        super(Loss_Reconstruction, self).__init__()
        self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, xs, xr):
        loss = self.loss(xs, xr)
        return loss


#
# class Loss_Konw_Guide(nn.Module):
#
#     def __init__(self):
#         super(Loss_Konw_Guide, self).__init__()
#
#     def forward(self,  proto,feature):
#         feature = F.normalize(feature, dim=1)
#
#         proto = F.normalize(proto, dim=1)
#         feature[torch.isinf(feature)] = 0
#         feature[torch.isnan(feature)] = 0
#         proto[torch.isinf(proto)] = 0
#         proto[torch.isnan(proto)] = 0
#         tau = 1
#         sim_matrix = torch.matmul(feature, proto.T) / tau
#         log_softmax_sim_matrix = F.log_softmax(sim_matrix, dim=1)
#         loss = -log_softmax_sim_matrix.mean()
#
#         return loss#, sim_matrix

class Loss_Konw_Guide(nn.Module):
    """
    知识引导损失：使用欧氏距离作为余弦相似度的自适应加权项
    距离越近，权重越大，强调局部几何结构
    """

    def __init__(self):
        super(Loss_Konw_Guide, self).__init__()

    def forward(self, proto, feature):

        feature = F.normalize(feature, dim=1)
        proto = F.normalize(proto, dim=1)
        feature = torch.nan_to_num(feature, nan=0.0, posinf=1.0, neginf=-1.0)
        proto = torch.nan_to_num(proto, nan=0.0, posinf=1.0, neginf=-1.0)

        cos_sim_matrix = torch.matmul(feature, proto.T)
        Nor_cos = F.softmax(cos_sim_matrix, dim=1)

        # 计算欧氏距离矩阵
        euclidean_dist = torch.cdist(feature, proto, p=2)  # [batch_size, n_prototypes]
        # 高斯权重：exp(-d^2/scale)
        distance_weights2 = torch.exp(-euclidean_dist.pow(2) )

        # log_softmax_sim_matrix = F.log_softmax(cos_sim_matrix, dim=1)

        log_distance_weights2 = torch.log(distance_weights2)
        lossen = log_distance_weights2 * Nor_cos
        # 计算损失
        loss = -lossen.mean()

        return loss

class Normalized_Mutual_Information(nn.Module):
    def __init__(self, bins=256):
        super(Normalized_Mutual_Information, self).__init__()
        self.bins = bins
        self.epsilon = 1e-8

    def forward(self, X1, X2):
        assert X1.shape == X2.shape, "输入张量形状必须相同"

        # 获取输入形状
        if X1.dim() == 2:
            batch_size, feature_dim = X1.shape
            height, width = 1, feature_dim  # 当作1D信号处理
        elif X1.dim() == 4:
            batch_size, channels, height, width = X1.shape
            feature_dim = channels * height * width
        else:
            raise ValueError("输入张量维度必须是2D或4D")

        # 展平处理，确保是一维数据
        X1_flat = X1.view(batch_size, -1)
        X2_flat = X2.view(batch_size, -1)
        feature_dim = X1_flat.shape[1]

        # 数据归一化到 [0, 1]
        X1_normalized = (X1_flat - X1_flat.min(dim=1, keepdim=True)[0]) / \
                        (X1_flat.max(dim=1, keepdim=True)[0] - X1_flat.min(dim=1, keepdim=True)[0] + self.epsilon)
        X2_normalized = (X2_flat - X2_flat.min(dim=1, keepdim=True)[0]) / \
                        (X2_flat.max(dim=1, keepdim=True)[0] - X2_flat.min(dim=1, keepdim=True)[0] + self.epsilon)

        # 离散化到bins个区间
        X1_discrete = (X1_normalized * (self.bins - 1)).long()
        X2_discrete = (X2_normalized * (self.bins - 1)).long()

        # 初始化联合概率矩阵
        joint_probs = torch.zeros(batch_size, self.bins, self.bins, device=X1.device)

        for i in range(batch_size):
            # 确保是一维向量
            indices_1d = X1_discrete[i] * self.bins + X2_discrete[i]

            # 展平并确保是一维
            indices_flat = indices_1d.view(-1)

            # 使用bincount计算直方图
            hist_1d = torch.bincount(indices_flat, minlength=self.bins * self.bins)

            # 重塑为二维直方图
            joint_probs[i] = hist_1d.view(self.bins, self.bins).float()

        # 归一化得到联合概率
        joint_probs = joint_probs / feature_dim

        # 计算边缘概率
        marginal_X1 = joint_probs.sum(dim=2)  # p(X1)
        marginal_X2 = joint_probs.sum(dim=1)  # p(X2)

        # 计算互信息
        p_x_p_y = marginal_X1.unsqueeze(2) * marginal_X2.unsqueeze(1)

        # 避免数值问题
        mask = (joint_probs > 0) & (p_x_p_y > 0)
        mi_terms = joint_probs * (torch.log(joint_probs + self.epsilon) -
                                  torch.log(p_x_p_y + self.epsilon))
        mi_terms = mi_terms * mask.float()
        mi = mi_terms.sum(dim=(1, 2))

        # 计算熵
        entropy_X1 = -torch.sum(marginal_X1 * torch.log(marginal_X1 + self.epsilon), dim=1)
        entropy_X2 = -torch.sum(marginal_X2 * torch.log(marginal_X2 + self.epsilon), dim=1)

        # 计算归一化互信息
        nmi = (2.0 * mi) / (entropy_X1 + entropy_X2 + self.epsilon)

        return nmi.mean()
