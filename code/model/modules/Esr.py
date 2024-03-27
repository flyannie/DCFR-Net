import torch.nn as nn
import torch.nn.functional as F
from utils.utils import MeanShift, ResBlock


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)


class Esr(nn.Module):
    def __init__(self, n_colors=3, n_resblocks=16, n_feats=64, res_scale=1, scale=2, no_upsampling=False, rgb_range=1, conv=default_conv):
        super(Esr, self).__init__()
        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3
        act = nn.ReLU(True)
        self.no_upsampling = no_upsampling
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)
        m_head = [conv(n_colors, n_feats, kernel_size)]
        m_body = [ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale) for _ in range(n_resblocks)]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        if self.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = n_colors
            m_tail = [conv(n_feats, n_colors, kernel_size)]
            self.tail = nn.Sequential(*m_tail)

    def forward(self, x, shape):
        x = self.head(x)
        res = self.body(x) 
        res += x
        if self.no_upsampling:
            x = res
            print("EDSR_no_up", x.shape)
        else:
            res = F.interpolate(res, shape)
            x = self.tail(res)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))






