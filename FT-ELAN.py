import torch
import torch.nn as nn
import math

from ..builder import BACKBONES


def autopad(k, p=None, d=1):  
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


class Focus(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0):
        super().__init__()
        self.conv = Conv(in_channels * 4, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


class Fourier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        x: (B, c, h, w)
        """
        # x = x.flatten(2)
        _, _, h, w = x.shape
        x = x.flatten(2)
        x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        x = torch.reshape(x, shape=(x.shape[0], x.shape[1], h, w))

        return x


class GConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=4):
        super().__init__()
        self.gconv = Conv(in_channels, in_channels, k=3, s=1, g=n_groups)
        self.conv = Conv(in_channels, out_channels, k=1)

    def forward(self, x):
        # for training/inference
        x = self.gconv(x)
        x = self.conv(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        self.fourier = Fourier()
        self.conv = Conv(c1, c2, k[0], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.conv(self.fourier(x)) if self.add else self.conv(self.fourier(x))   


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c      = int(c2 * e) 
        self.cv1    = Conv(c1, 2 * self.c, 1, 1)
        self.cv2    = Conv((2 + n) * self.c, c2, 1)
        self.m      = nn.ModuleList(Bottleneck(self.c, self.c, shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_          = c1 // 2
        self.cv1    = Conv(c1, c_, 1, 1)
        self.cv2    = Conv(c_ * 4, c2, 1, 1)
        self.m      = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


@BACKBONES.register_module()
class MY_BACKBONE(nn.Module):
    def __init__(self, base_channels, base_depth, deep_mul):
        super().__init__()
        # 3, 640, 640 => 32, 640, 640 => 64, 320, 320
        # self.stem = Conv(3, base_channels, 3, 2)
        # self.stem = nn.Sequential(
        #     Conv(3, base_channels, 3, 2),
        #     Fourier(),
        # )
        self.stem = Focus(3, base_channels, 3, 1, 1)

        # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
        self.dark2 = nn.Sequential(
            C2f(base_channels, base_channels, base_depth, True),
            Conv(base_channels, base_channels * 2, 3, 2),
        )
        # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
        self.dark3 = nn.Sequential(
            C2f(base_channels * 2, base_channels * 2, base_depth * 2, True),
            Conv(base_channels * 2, base_channels * 4, 3, 2),
        )
        # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
        self.dark4 = nn.Sequential(
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
            Conv(base_channels * 4, base_channels * 8, 3, 2),
        )
        # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
        self.dark5 = nn.Sequential(
            # Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)
        )
        

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        x = self.dark3(x)
        feat1 = x
        x = self.dark4(x)
        feat2 = x
        x = self.dark5(x)
        feat3 = x
        return tuple([feat1, feat2, feat3])