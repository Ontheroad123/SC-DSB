import torch
import torchvision
import sys

from torchvision.utils import save_image
import torch
import torch.nn as nn
import torchvision.models as models


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.device = torch.device(f'cuda')

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, x_t, feature_layers=[0, 1, 2, 3], style_layers=[]):

        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)

     
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
       
        loss = 0.0
        x = input
        y = target

        
        for i, block in enumerate(self.blocks):
            x = x.to(torch.float16)
            y = y.to(torch.float16)
            block = block.to(self.device).to(torch.float16)
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


def total_variation_loss(image):
    """计算图像的总变差损失。

    Args:
        image: 一个形状为 [batch_size, channels, height, width] 的图像张量。

    Returns:
        图像的总变差损失。
    """

    # 计算图像的水平和垂直梯度
    h_grad = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    v_grad = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])

    # 计算总变差损失
    loss = torch.mean(h_grad) + torch.mean(v_grad)

    return loss



class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self, target, output):
        """计算两幅图像的Style Loss。

        Args:
            style_gram:  风格图像的Gram矩阵。
            target_gram:  目标图像的Gram矩阵。

        Returns:
            两幅图像的Style Loss。
        """
        style_gram = gram_matrix(output)
        target_gram = gram_matrix(target)
        loss = nn.MSELoss()(style_gram, target_gram)
        return loss

    def gram_matrix(self, x):
        # x 的形状为 [batch_size, C, H, W]
        (batch_size, C, H, W) = x.size()
        F = x.view(batch_size, C, H*W)
        G = F.transpose(1,2) @ F
        return G


class CurvatureLoss(nn.Module):
    def __init__(self, p=2):
        super(CurvatureLoss, self).__init__()
        self.p = p

    def forward(self, input, target):
        # 计算两个形状的曲率 K1 和 K2
        K1 = compute_curvature(input)
        K2 = compute_curvature(target)

        # 计算曲率差异
        curvature_diff = torch.abs(K1 - K2)

        # 计算曲率损失
        curvature_loss = torch.norm(curvature_diff, p=self.p)

        return curvature_loss

def compute_curvature(shape):
    # 使用你选择的曲率计算方法计算曲率
    # 例如，可以使用主曲率计算方法：
    K = compute_principal_curvature(shape)
    return K
# import numpy as np 
# import torch
# # 假设input和target是两个图像张量
# perceptual_loss = VGGPerceptualLoss()
# a = np.zeros((16,4,16,16))
# loss = perceptual_loss(torch.zeros((16,4,16,16)), torch.zeros((16,4,16,16)))
# print(loss)