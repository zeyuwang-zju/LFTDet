import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule

from ..builder import NECKS


def autopad(k, p=None, d=1):  
    # kernel, padding, dilation
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SiLU(nn.Module):  
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Conv(nn.Module):
    default_act = SiLU() 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class BasicBlock(nn.Module):
    # expansion = 1

    def __init__(self, filter_in, filter_out):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(filter_in, filter_out, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filter_out, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(filter_out, filter_out, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filter_out, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )

    def forward(self, x):
        x = self.upsample(x)

        return x


class SimAM(nn.Module):
    def __init__(self, lambda_=1e-4):
        super(SimAM, self).__init__()

        self.lambda_ = lambda_

    def forward(self, x):
        n = x.shape[2] * x.shape[3] - 1
        d = (x - torch.mean(x, dim=[2,3], keepdim=True)).pow(2)
        v = torch.sum(d, dim=[2,3], keepdim=True) / n
        E_inv = d / (4 * (v + self.lambda_)) + 0.5

        return x * torch.sigmoid(E_inv)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Downsample, self).__init__()

        self.downsample = nn.Sequential(
            Conv(in_channels, out_channels, scale_factor, scale_factor, 0)
        )

    def forward(self, x):
        x = self.downsample(x)

        return x


class ASFF_2(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_2, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1, 0)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1, 0)

        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1, 1)

    def forward(self, input1, input2):
        """
        input1, input2: (inter_dim, h, w)
        output: (inter_dim, h, w)
        """
        level_1_weight_v = self.weight_level_1(input1)  # (compress_c, h, w)
        level_2_weight_v = self.weight_level_2(input2)  # (compress_c, h, w)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)    # (compress_c * 2, h, w)

        levels_weight = self.weight_levels(levels_weight_v)     # (2, h, w)
        levels_weight = F.softmax(levels_weight, dim=1)         # (2, h, w)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :]    # (inter_dim, h, w)

        out = self.conv(fused_out_reduced)      # (inter_dim, h, w)

        return out


class ASFF_3(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_3, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1, 0)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1, 0)
        self.weight_level_3 = Conv(self.inter_dim, compress_c, 1, 1, 0)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1, 1)

    def forward(self, input1, input2, input3):
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)

        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :] + \
                            input3 * levels_weight[:, 2:, :, :]

        out = self.conv(fused_out_reduced)

        return out


@NECKS.register_module()
class My_Neck(BaseModule):
    def __init__(self,
                 in_channels=[256, 512, 1024],
                 out_channels=256,
                 compress_ratio=8,
                 num_blocks=4):
        super(My_Neck, self).__init__()

        self.in_channels = in_channels

        self.simam = SimAM()

        self.conv0 = Conv(in_channels[0], in_channels[0] // compress_ratio, 1, 1, 0)
        self.conv1 = Conv(in_channels[1], in_channels[1] // compress_ratio, 1, 1, 0)
        self.conv2 = Conv(in_channels[2], in_channels[2] // compress_ratio, 1, 1, 0)


        self.blocks_scalezero1 = Conv(in_channels[0] // compress_ratio, in_channels[0] // compress_ratio, 1, 1, 0)
        self.blocks_scaleone1 = Conv(in_channels[1] // compress_ratio, in_channels[1] // compress_ratio, 1, 1, 0)
        self.blocks_scaletwo1 = Conv(in_channels[2] // compress_ratio, in_channels[2] // compress_ratio, 1, 1, 0)

        self.downsample_scalezero1_2 = Downsample(in_channels[0] // compress_ratio, in_channels[1] // compress_ratio, scale_factor=2)
        self.upsample_scaleone1_2 = Upsample(in_channels[1] // compress_ratio, in_channels[0] // compress_ratio, scale_factor=2)

        self.asff_scalezero1 = ASFF_2(inter_dim = in_channels[0] // compress_ratio)
        self.asff_scaleone1 = ASFF_2(inter_dim = in_channels[1] // compress_ratio)

        self.blocks_scalezero2 = nn.Sequential(*[
            BasicBlock(in_channels[0] // compress_ratio, in_channels[0] // compress_ratio) for _ in range(num_blocks)
        ])

        self.blocks_scaleone2 = nn.Sequential(*[
            BasicBlock(in_channels[1] // compress_ratio, in_channels[1] // compress_ratio) for _ in range(num_blocks)
        ])

        self.downsample_scalezero2_2 = Downsample(in_channels[0] // compress_ratio, in_channels[1] // compress_ratio, scale_factor=2)
        self.downsample_scalezero2_4 = Downsample(in_channels[0] // compress_ratio, in_channels[2] // compress_ratio, scale_factor=4)
        self.downsample_scaleone2_2 = Downsample(in_channels[1] // compress_ratio, in_channels[2] // compress_ratio, scale_factor=2)
        self.upsample_scaleone2_2 = Upsample(in_channels[1] // compress_ratio, in_channels[0] // compress_ratio, scale_factor=2)
        self.upsample_scaletwo2_2 = Upsample(in_channels[2] // compress_ratio, in_channels[1] // compress_ratio, scale_factor=2)
        self.upsample_scaletwo2_4 = Upsample(in_channels[2] // compress_ratio, in_channels[0] // compress_ratio, scale_factor=4)

        self.asff_scalezero2 = ASFF_3(inter_dim=in_channels[0] // compress_ratio)
        self.asff_scaleone2 = ASFF_3(inter_dim=in_channels[1] // compress_ratio)
        self.asff_scaletwo2 = ASFF_3(inter_dim=in_channels[2] // compress_ratio)

        self.blocks_scalezero3 = nn.Sequential(*[
            BasicBlock(in_channels[0] // compress_ratio, in_channels[0] // compress_ratio) for _ in range(num_blocks)
        ])

        self.blocks_scaleone3 = nn.Sequential(*[
            BasicBlock(in_channels[1] // compress_ratio, in_channels[1] // compress_ratio) for _ in range(num_blocks)
        ])

        self.blocks_scaletwo3 = nn.Sequential(*[
            BasicBlock(in_channels[2] // compress_ratio, in_channels[2] // compress_ratio) for _ in range(num_blocks)
        ])

        self.conv00 = Conv(in_channels[0] // compress_ratio, out_channels, 1, 1, 0)
        self.conv11 = Conv(in_channels[1] // compress_ratio, out_channels, 1, 1, 0)
        self.conv22 = Conv(in_channels[2] // compress_ratio, out_channels, 1, 1, 0)


    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.
            (C0, 80, 80), (C1, 40, 40), (C2, 20, 20)

        Returns:
            tuple[Tensor]: output features.
        """
        assert len(inputs) == len(self.in_channels)
        x0, x1, x2 = inputs

        x0 = self.conv0(x0)         # (C0 // compress_ratio, 80, 80)
        x1 = self.conv1(x1)         # (C1 // compress_ratio, 40, 40)
        x2 = self.conv2(x2)         # (C2 // compress_ratio, 20, 20)

        x0 = self.blocks_scalezero1(x0)        # (C0 // compress_ratio, 80, 80)
        x1 = self.blocks_scaleone1(x1)         # (C1 // compress_ratio, 40, 40)
        x2 = self.blocks_scaletwo1(x2)         # (C2 // compress_ratio, 20, 20)

        scalezero = self.asff_scalezero1(x0, self.upsample_scaleone1_2(x1))     # (C0 // compress_ratio, 80, 80)
        scaleone = self.asff_scaleone1(self.downsample_scalezero1_2(x0), x1)    # (C1 // compress_ratio, 40, 40)

        scalezero = self.simam(scalezero)
        scaleone = self.simam(scaleone)

        x0 = self.blocks_scalezero2(scalezero)      # (C0 // compress_ratio, 80, 80)
        x1 = self.blocks_scaleone2(scaleone)        # (C1 // compress_ratio, 40, 40)

        scalezero = self.asff_scalezero2(x0, self.upsample_scaleone2_2(x1), self.upsample_scaletwo2_4(x2))      # (C0 // compress_ratio, 80, 80)
        scaleone = self.asff_scaleone2(self.downsample_scalezero2_2(x0), x1, self.upsample_scaletwo2_2(x2))     # (C1 // compress_ratio, 40, 40)
        scaletwo = self.asff_scaletwo2(self.downsample_scalezero2_4(x0), self.downsample_scaleone2_2(x1), x2)   # (C2 // compress_ratio, 20, 20)

        scalezero = self.simam(scalezero)
        scaleone = self.simam(scaleone)
        scaletwo = self.simam(scaletwo)

        x0 = self.blocks_scalezero3(scalezero)      # (C0 // compress_ratio, 80, 80)
        x1 = self.blocks_scaleone3(scaleone)        # (C1 // compress_ratio, 40, 40)
        x2 = self.blocks_scaletwo3(scaletwo)        # (C2 // compress_ratio, 20, 20)

        out0 = self.conv00(x0)
        out1 = self.conv11(x1)
        out2 = self.conv22(x2)

        return tuple([out0, out1, out2])