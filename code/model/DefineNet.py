import torch
from torch.nn import init
from .modules.IDN import IDN
from .modules.Esr import Esr
from .modules.Diffusion import Diffusion
from .ISDTDNet import ISDTDNet


def define_DCHFRnet(args):
    IDN_model = IDN(in_channel=6, out_channel=3, norm_groups=32, inner_channel=64, channel_mults=[1,2,4,8,8], attn_res=[32], res_blocks=2, dropout=0.2, image_size=args.base_size)
    encoder = Esr(n_resblocks=16, n_feats=64, res_scale=1, no_upsampling=False, rgb_range=1)
    DCHFR_net = Diffusion(encoder, denoise_fn=IDN_model, image_size=args.base_size, channels=3, conditional=True)
    init_weights(DCHFR_net, init_type='orthogonal')
    assert torch.cuda.is_available()
    return DCHFR_net

def define_trained_DCHFRnet(args):
    IDN_model = IDN(in_channel=6, out_channel=3, norm_groups=32, inner_channel=64, channel_mults=[1,2,4,8,8], attn_res=[32], res_blocks=2, dropout=0.2, image_size=args.base_size)
    encoder = Esr(n_resblocks=16, n_feats=64, res_scale=1, scale=4, no_upsampling=False, rgb_range=1)
    net = Diffusion(encoder, denoise_fn=IDN_model, image_size=args.base_size, channels=3, conditional=True)
    assert torch.cuda.is_available()
    return net

def define_ISDTDnet(args):
    ISDTD_net = ISDTDNet(args)
    assert torch.cuda.is_available()
    return ISDTD_net

def init_weights(net, init_type='orthogonal'):
    if init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)