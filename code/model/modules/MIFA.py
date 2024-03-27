import torch
import torch.nn as nn
from .INR_align import INR_align


class MIFA(nn.Module):

    def __init__(self):
        super(MIFA, self).__init__()
        self.INR_align = INR_align()
        norm_layer = nn.BatchNorm2d
        self.head = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=1), norm_layer(256), nn.ReLU(inplace=True))
        self.enc1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1), norm_layer(256), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), norm_layer(256), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1), norm_layer(256), nn.ReLU(inplace=True))

    def forward(self, x):
        x1, x2, x3, x4 = x
        aspp_out = self.head(x4)
        x1 = self.enc1(x1)
        x2 = self.enc2(x2)
        x3 = self.enc3(x3)
        context = []
        h, w = x1.shape[-2], x1.shape[-1]
        target_feat = [x1, x2, x3, aspp_out]
        for i, feat in enumerate(target_feat):
            context.append(self.INR_align(feat, size=[h, w], level=i+1))
        context = torch.cat(context, dim=-1).permute(0,2,1)
        res = self.INR_align(context, size=[h, w], after_cat=True)
        return res




