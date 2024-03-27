import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(input, target):
    smooth = 1e-5
    num = target.size(0)
    input = input.view(num, -1)
    target = target.view(num, -1)
    intersection = (input * target)
    dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    return dice 

class BCEDiceloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = torch.sigmoid(input)
        dice = dice_loss(input,target)
        bce = F.binary_cross_entropy(input, target)
        return 0.8 * bce + 0.2 * dice, bce, dice
