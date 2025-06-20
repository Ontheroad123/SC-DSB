import torch 
import torch.nn  as nn 
import torchvision.models  as models
from torchvision.models  import vgg11
import torch as th
import math
from nn import (
    SiLU,
    conv_nd,
    linear)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class NoiseNet(nn.Module):
    def __init__(self):
        super(NoiseNet, self).__init__()
        # 加载预训练的 VGG11 骨干网络
        self.backbone  = vgg11(pretrained=True)
        # 替换最后一层为全连接层
        self.backbone.classifier  = nn.Sequential(
            nn.ReLU(),
            nn.Linear(33280 , 256),  # 增加一个维度来处理时间步 t 
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid()  # 确保输出在0到1之间
        )
        model_channels = 128
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
 
    def forward(self, x, t):
        # 获取骨干网络的特征 
        features = self.backbone.features(x) 
        features = features.view(features.size(0),  -1)  # 展平特征 
        
        #t = t.unsqueeze(-1).expand(features.shape[0],  1)
        t = self.time_embed(timestep_embedding(t, 128))
        print(t.shape)
        features = torch.cat((features,  t), dim=1)
        print(features.shape)
        # 通过全连接层进行前向传播
        output = self.backbone.classifier(features) 
        return output
 
# 初始化模型 
# model = NoiseNet() 
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
# # 随机生成一个输入张量代表图像数据 
# x = torch.randn(1,  3, 256, 256)  
 
# # 进行前向传播 
# output = model(x, t = torch.randint(1,  100, (x.shape[0],)).float()) 
# print("Output shape:", output.shape, output)  