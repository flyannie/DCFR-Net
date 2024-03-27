import torch
import torch.nn as nn
from utils.utils import SpatialEncoding, gen_feat


class INR_align(nn.Module):
    def __init__(self, pos_dim=24, stride=1, require_grad=True):
        super(INR_align, self).__init__()
        self.pos_dim = pos_dim
        self.stride = stride
        norm_layer = nn.BatchNorm1d
        self.pos1 = SpatialEncoding(2, self.pos_dim, require_grad=require_grad)
        self.pos2 = SpatialEncoding(2, self.pos_dim, require_grad=require_grad)
        self.pos3 = SpatialEncoding(2, self.pos_dim, require_grad=require_grad)
        self.pos4 = SpatialEncoding(2, self.pos_dim, require_grad=require_grad)
        self.pos_dim += 2
        in_dim = 4 * (256 + self.pos_dim)
        self.inr = nn.Sequential(
            nn.Conv1d(in_dim, 512, 1), norm_layer(512), nn.ReLU(),
            nn.Conv1d(512, 256, 1), norm_layer(256), nn.ReLU(),
            nn.Conv1d(256, 256, 1), norm_layer(256), nn.ReLU(),
            nn.Conv1d(256, 3, 1) 
        )

    def forward(self, x, size, level=0, after_cat=False):
        h, w = size
        if not after_cat:
            rel_coord, q_feat = gen_feat(x, [h, w])
            rel_coord = eval('self.pos' + str(level))(rel_coord)
            x = torch.cat([rel_coord, q_feat], dim=-1)
        else:
            x = self.inr(x)
            x = x.view(x.shape[0], -1, h, w)
        return x


