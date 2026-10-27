import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim):
        """
        初始化 Encoder
        :param input_dim: 输入数据的维度 (例如: 图片像素总数或特征数量)
        """
        super(Encoder, self).__init__()

        self.net = nn.Sequential(
            # 第一层: Input -> 1024
            nn.Linear(input_dim, 1024),
            nn.ReLU(),

            # 第二层: 1024 -> 512
            nn.Linear(1024, 512),
            nn.ReLU(),

            # 第三层 (输出层/Latent): 512 -> 64
            # 通常 Encoder 的最后一层不加激活函数，以便保留完整的特征空间分布，
            # 如果你需要输出非负特征，可以在这里也加上 nn.ReLU()
            nn.Linear(512, 64)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, output_dim):
        """
        初始化 Decoder
        :param output_dim: 还原后的输出维度 (必须与 Encoder 的输入维度一致)
        """
        super(Decoder, self).__init__()

        self.net = nn.Sequential(
            # 第一层: Latent(64) -> 512
            nn.Linear(64, 512),
            nn.ReLU(),

            # 第二层: 512 -> 1024
            nn.Linear(512, 1024),
            nn.ReLU(),

            # 第三层 (输出层): 1024 -> 原始维度
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        return self.net(x)