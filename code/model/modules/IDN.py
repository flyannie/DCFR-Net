import torch
from torch import nn
from utils.utils import PositionalEncoding, Swish, ResnetBlocWithAttn, Downsample, Block, default, exists
from .Style import StyleLayer, EqualLinear
from .INR_sr import INR_sr
from einops import rearrange


class IDN(nn.Module):
    def __init__(self,in_channel=6,out_channel=3,inner_channel=64,norm_groups=32,channel_mults=(1, 2, 4, 8, 8),attn_res=(32),res_blocks=2,dropout=0.2,image_size=256):
        super().__init__()
        noise_level_channel = inner_channel
        self.noise_level_mlp = nn.Sequential(PositionalEncoding(inner_channel), nn.Linear(inner_channel, inner_channel * 4), Swish(), nn.Linear(inner_channel * 4, inner_channel))
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        self.conv_body_first = StyleLayer(3, pre_channel, 3, bias=True, activate=True)
        self.conv_body_down = nn.ModuleList()
        self.condition_scale1 = nn.ModuleList()
        self.condition_scale2 = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            self.conv_body_down.append(StyleLayer(pre_channel, channel_mult, 3, downsample=True))
            self.condition_scale1.append(EqualLinear(1, channel_mult, bias=True, bias_init_val=1, activation=None))
            self.condition_scale2.append(EqualLinear(1, channel_mult, bias=True, bias_init_val=1, activation=None))
            self.condition_shift.append(StyleLayer(pre_channel, channel_mult, 3, bias=True, activate=False))
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)
        self.final_down1 = StyleLayer(512, 512, 3, downsample=False)
        self.final_down2 = StyleLayer(512, 256, 3, downsample=True)
        self.num_latent, self.num_style_feat = 4, 512
        self.final_linear = EqualLinear(2 *2 * 256, self.num_style_feat * self.num_latent, bias=True, activation='fused_lrelu')
        self.final_styleconv = StyleLayer(512, 512, 3)
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=False)
        ])
        ups = []
        for ind in reversed(range(num_mults)): 
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(ResnetBlocWithAttn(pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(INR_sr(pre_channel))
                now_res = now_res * 2
        self.ups = nn.ModuleList(ups)
        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, lr, scaler, time):
        t = self.noise_level_mlp(time) if exists(self.noise_level_mlp) else None 
        feat = self.conv_body_first(lr)
        scales1, scales2, shifts = [], [], []
        scale1 = self.condition_scale1[0](scaler)
        scales1.append(scale1.clone())
        scale2 = self.condition_scale2[0](scaler)
        scales2.append(scale2.clone())
        shift = self.condition_shift[0](feat)
        shifts.append(shift.clone())
        j = 1
        for i in range(len(self.conv_body_down)):
            feat = self.conv_body_down[i](feat) 
            if j < len(self.condition_scale1) : 
                scale1 = self.condition_scale1[j](scaler)
                scales1.append(scale1.clone())
                scale2 = self.condition_scale2[j](scaler)
                scales2.append(scale2.clone())
                shift = self.condition_shift[j](feat)
                shifts.append(shift.clone())
                j += 1 
        feats = []
        for i,layer in enumerate(self.downs):
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t) 
            else:
                x = layer(x)
            feats.append(x)
        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for i, layer in enumerate(self.ups):
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x, feats[-1].shape[2:], scales1.pop(), scales2.pop(), shifts.pop())
                x = rearrange(x, 'b (h w) c -> b c h w', h=feats[-1].shape[-1])
        return self.final_conv(x)

