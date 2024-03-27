import torch.nn as nn
from .modules.SFFI import SFFI
from .modules.MIFA import MIFA
from torch.nn import functional as F
from .modules.half_IDN import half_IDN


class ISDTDNet(nn.Module):

    def __init__(self, args):
        super(ISDTDNet, self).__init__()
        self.encoder = half_IDN(in_channel=3, norm_groups=32, inner_channel=64, channel_mults=[1,2,4,8,16], attn_res=[32], res_blocks=2, dropout=0.2, image_size=args.base_size)
        self.MIFA = MIFA()
        self.SFFI = SFFI(input_dim=3, hidden_dim=64)
        
    def forward(self, feat, img):
        _, _, h, w = feat.size()
        feat = self.encoder(feat,img)
        out = self.MIFA(feat)
        resize_out = F.upsample(input=out, size=(h, w), mode='bilinear', align_corners=True)
        mask = self.SFFI(resize_out)
        return mask








