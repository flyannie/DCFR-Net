import torch
from torch import nn
from .Style import StyleLayer
from utils.utils import Uhalf_ResnetBlocWithAttn, Downsample

class half_IDN(nn.Module):
    def __init__(self, in_channel=6, inner_channel=64, norm_groups=32, channel_mults=(1, 2, 4, 8, 16), attn_res=(32), res_blocks=2, dropout=0.2, image_size=256):
        super().__init__()
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        self.conv_body_first = StyleLayer(3, pre_channel, 3, bias=True, activate=True)
        self.conv_body_down = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            self.conv_body_down.append(StyleLayer(pre_channel, channel_mult, 3, downsample=True))
            self.condition_shift.append(StyleLayer(pre_channel, channel_mult, 3, bias=True, activate=False))
            for _ in range(0, res_blocks):
                downs.append(Uhalf_ResnetBlocWithAttn(pre_channel, channel_mult, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

    def forward(self, feat, img):
        feat = self.conv_body_first(feat)
        # g0 = self.condition_shift[0](feat)
        feat = self.conv_body_down[0](feat)
        g1 = self.condition_shift[1](feat)
        feat = self.conv_body_down[1](feat)
        g2 = self.condition_shift[2](feat)
        feat = self.conv_body_down[2](feat)
        g3 = self.condition_shift[3](feat)
        feat = self.conv_body_down[3](feat)
        g4 = self.condition_shift[4](feat)
        feat = self.downs[0](img)
        f0_1 = self.downs[1](feat)
        f0 = self.downs[2](f0_1)
        f1_1 = self.downs[3](f0)
        f1_2 = self.downs[4](f1_1)
        f1 = self.downs[5](f1_2)
        f2_1 = self.downs[6](f1)
        f2_2 = self.downs[7](f2_1)
        f2 = self.downs[8](f2_2)
        f3_1 = self.downs[9](f2)
        f3_2 = self.downs[10](f3_1)
        f3 = self.downs[11](f3_2)
        f4_1 = self.downs[12](f3)
        f4_2 = self.downs[13](f4_1)
        f4 = self.downs[14](f4_2)
        m1 = torch.cat([g1, f1], dim=1)
        m2 = torch.cat([g2, f2], dim=1)
        m3 = torch.cat([g3, f3], dim=1)
        m4 = torch.cat([g4, f4], dim=1)
        return [m1, m2, m3, m4]

