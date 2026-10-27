import torch.nn as nn
from torch.nn.functional import normalize
import torch.nn.functional as F
import torch


class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self, input_dim, feature_dim, activation='relu', batchnorm=True):
        super(Autoencoder, self).__init__()
        self._activation = activation
        self._batchnorm = batchnorm

        # Encoder definition
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
        )

        # Decoder definition
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, input_dim].

        Returns:
            z: Latent representation of shape [batch_size, feature_dim].
            xr: Reconstructed input of shape [batch_size, input_dim].
        """
        z = self.encoder(x)
        xr = self.decoder(z)
        return z, xr




class Fast_DynamicFusion(nn.Module):
    """
    快速NMI动态融合模块 - 计算复杂度大幅降低
    """

    def __init__(self, feature_dim=256, hidden_dim=64, temperature=0.1, epsilon=1e-8, similarity_alpha=0.3):
        super(Fast_DynamicFusion, self).__init__()

        self.feature_dim = feature_dim
        self.temperature = temperature
        self.epsilon = epsilon
        self.similarity_alpha = similarity_alpha

        # 轻量级相似度估计网络（替代复杂的NMI计算）
        self.similarity_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),  # 直接输出每个特征的相似度估计
            nn.Sigmoid()  # 输出[0,1]范围的相似度
        )

        # 轻量级融合权重网络
        self.fusion_net = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),  # 输入: X1, X2, 相似度
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim * 2),  # 输出每个特征的融合权重
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def fast_similarity_estimation(self, X1, X2):
        """
        快速相似度估计 - 替代复杂的NMI计算
        使用相关性+神经网络快速估计特征相似度
        """
        batch_size = X1.shape[0]

        # 方法1: 快速相关性计算（比NMI快得多）
        with torch.no_grad():
            # 计算每个特征的皮尔逊相关系数
            X1_centered = X1 - X1.mean(dim=0, keepdim=True)
            X2_centered = X2 - X2.mean(dim=0, keepdim=True)

            # 快速相关性计算
            correlation = (X1_centered * X2_centered).sum(dim=0) / (
                    torch.norm(X1_centered, dim=0) * torch.norm(X2_centered, dim=0) + self.epsilon
            )

            # 将相关性映射到[0,1]范围
            fast_similarity = (correlation + 1) / 2  # [-1,1] -> [0,1]

        # 方法2: 神经网络精修相似度（可选，可关闭加速）
        if self.training:
            # 只在训练时使用神经网络精修
            concatenated = torch.cat([X1, X2], dim=1)
            net_similarity = self.similarity_net(concatenated).mean(dim=0)

            # 结合快速估计和神经网络估计
            similarity = self.similarity_alpha * fast_similarity + (1 - self.similarity_alpha) * net_similarity
        else:
            # 推理时只用快速估计
            similarity = fast_similarity

        return similarity

    def efficient_weight_generation(self, X1, X2, similarities):
        """
        高效权重生成 - 批量处理所有特征
        """
        batch_size = X1.shape[0]

        # 扩展相似度到batch维度
        similarities_expanded = similarities.unsqueeze(0).expand(batch_size, -1)  # [batch, features]

        # 准备融合网络的输入：X1, X2, 相似度
        fusion_input = torch.cat([X1, X2, similarities_expanded], dim=1)  # [batch, features*3]

        # 一次性生成所有特征的权重
        raw_weights = self.fusion_net(fusion_input)  # [batch, features*2]
        raw_weights = raw_weights.reshape(batch_size, self.feature_dim, 2)  # [batch, features, 2]

        # 应用softmax确保权重和为1
        weights = F.softmax(raw_weights / self.temperature, dim=2)  # [batch, features, 2]

        # 转置为 [batch, 2, features] 格式
        weights = weights.transpose(1, 2)

        return weights

    def forward(self, X1, X2, return_analysis=False):
        """
        高效前向传播
        """
        # 1. 快速相似度估计（替代复杂的NMI计算）
        similarities = self.fast_similarity_estimation(X1, X2)  # [features]

        # 2. 高效权重生成
        weights = self.efficient_weight_generation(X1, X2, similarities)  # [batch, 2, features]

        # 3. 应用融合
        weighted_X1 = X1 * weights[:, 0, :]  # [batch, features]
        weighted_X2 = X2 * weights[:, 1, :]  # [batch, features]
        fused = weighted_X1 + weighted_X2

        if return_analysis:
            analysis = {
                'similarities': similarities,
                'weights': weights,
                'mean_similarity': similarities.mean(),
                'weight_stats': {
                    'view1_mean': weights[:, 0, :].mean(dim=1),
                    'view2_mean': weights[:, 1, :].mean(dim=1),
                }
            }
            return fused, weights, analysis
        else:
            return fused, weights












