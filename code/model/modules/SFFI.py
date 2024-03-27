import torch
import torch.nn as nn
import torch.fft as fft
from einops import rearrange

class NonLocalAttention(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feat) -> torch.Tensor:
        b, c, h, w = feat.shape
        out_feat = self.conv(feat)
        out_feat = rearrange(out_feat, 'b c h w -> b (h w) c')
        out_feat = torch.unsqueeze(out_feat, -1)
        out_feat = self.softmax(out_feat)
        out_feat = torch.squeeze(out_feat, -1)
        identity = rearrange(feat, 'b c h w -> b c (h w)')
        out_feat = torch.matmul(identity, out_feat)
        out_feat = torch.unsqueeze(out_feat, -1)
        return out_feat


class NonLocalAttentionBlock(nn.Module):

    def __init__(self, in_channels) -> None:
        super().__init__()
        self.nonlocal_attention = NonLocalAttention(in_channels)
        self.global_transform = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, feat):
        out_feat = self.nonlocal_attention(feat)
        out_feat = self.global_transform(out_feat)
        return feat + out_feat


class SpectralTransformer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels * 2, 3, padding=1)
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        out_feat = fft.rfft2(feat)
        out_feat = torch.cat([out_feat.real, out_feat.imag], dim=1)
        out_feat = self.conv(out_feat)
        out_feat = self.lrelu(out_feat)
        c = out_feat.shape[1]
        out_feat = torch.complex(out_feat[:, : c // 2], out_feat[:, c // 2 :])
        out_feat = fft.irfft2(out_feat)
        return out_feat


class FourierConvolutionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.half_channels = in_channels // 2
        self.func_g_to_g = SpectralTransformer(self.half_channels)
        self.func_g_to_l = nn.Sequential(
            nn.Conv2d(self.half_channels, self.half_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.func_l_to_g = nn.Sequential(
            nn.Conv2d(self.half_channels, self.half_channels, kernel_size=1),
            NonLocalAttentionBlock(self.half_channels),
        )
        self.func_l_to_l = nn.Sequential(
            nn.Conv2d(self.half_channels, self.half_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        global_feat = feat[:, self.half_channels :]
        local_feat = feat[:, : self.half_channels]
        out_global_feat = self.func_l_to_g(local_feat) + self.func_g_to_g(global_feat)
        out_local_feat = self.func_g_to_l(global_feat) + self.func_l_to_l(local_feat)
        return torch.cat([out_global_feat, out_local_feat], 1)


class SFFI(nn.Module):
    def __init__(self, input_dim=3,hidden_dim = 64) -> None:
        super().__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 1),
            FourierConvolutionBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            FourierConvolutionBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )
        self.second_block = nn.Sequential(
            FourierConvolutionBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            FourierConvolutionBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )
        self.third_block = nn.Sequential(
            FourierConvolutionBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            FourierConvolutionBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            FourierConvolutionBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            FourierConvolutionBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )
        self.fourth_block = nn.Sequential(
            FourierConvolutionBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim//4,  kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=hidden_dim//4),
            nn.ReLU(),# nn.Sigmoid,
            nn.Conv2d( hidden_dim//4, 1,  kernel_size=1, padding=0, bias=False),
        )
    def forward(self, feat):
        first_feat = self.first_block(feat)
        second_feat = first_feat + self.second_block(first_feat)
        third_feat = second_feat + self.third_block(second_feat)
        result=self.fourth_block(third_feat)
        return result


