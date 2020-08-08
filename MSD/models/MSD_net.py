import torch

from torch import nn


class MSDnet(nn.Module):

    def __init__(self):
        super().__init__()

        self.initial_reduce = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # layer 1 outputs
        self.h11_stride = Layer1Scale(in_ch=3, out_ch=6, stride=1)
        self.h21_stride = Layer1Scale(in_ch=6, out_ch=12, stride=2)
        self.h31_stride = Layer1Scale(in_ch=12, out_ch=24, stride=2)

        self.h11_regular = Layer1Scale(in_ch=3, out_ch=6, stride=1)
        self.h21_regular = Layer1Scale(in_ch=6, out_ch=12, stride=1)
        self.h31_regular = Layer1Scale(in_ch=12, out_ch=24, stride=1)

        # layer 2
        self.h12 = LayerNScale(in_ch=6, mid_ch=6, out_ch=6, out_stride=1)
        self.h22 = LayerNScale(in_ch=18, mid_ch=12, out_ch=12, out_stride=)
        self.h32 = LayerNScale(in_ch=, mid_ch=, out_ch=, out_stride=)



    def forward(self, x):
        x = self.initial_reduce(x)


        return x


class Layer1Scale(nn.Module):

    def __init__(self, in_ch, out_ch, stride):
        super().__init__()

        self.stride_down = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        self.stride_down(x)
        return x


class LayerNScale(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, out_stride):
        super().__init__()

        self.stride_down = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_ch, out_channels=out_ch, kernel_size=3, stride=out_stride),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        self.stride_down(x)
        return x
