from MSD.settings.config import NUM_CLASSES

import torch
from torch import nn


class MSDnet(nn.Module):

    def __init__(self):
        super().__init__()

        self.initial_reduce = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # layer 1
        self.h11_stride = Layer1Scale(in_ch=3, out_ch=6, stride=2, pad=1)
        self.h21_stride = Layer1Scale(in_ch=6, out_ch=12, stride=2, pad=1)

        # self.h11_regular = Layer1Scale(in_ch=3, out_ch=6, stride=1, pad=1)
        # self.h21_regular = Layer1Scale(in_ch=6, out_ch=12, stride=1, pad=1)
        self.h31_regular = Layer1Scale(in_ch=12, out_ch=24, stride=1, pad=1)

        # layer 2
        # self.h12 = LayerNScale(in_ch=6, mid_ch=6, out_ch=6, stride=0)
        # self.h22 = LayerNScale(in_ch=18, mid_ch=12, out_ch=12, stride=0)
        # self.h32 = LayerNScale(in_ch=36, mid_ch=24, out_ch=24, stride=0)


        # classifier after layer 2
        self.classifier_l2 = Classifier(in_ch=36, mid_ch=36, out_ch=36)

    def forward(self, x):
        x_reduced = self.initial_reduce(x)
        # layer 1
        h11_s_out = self.h11_stride(x_reduced)
        h21_s_out = self.h21_stride(h11_s_out)
        h31_r_out = self.h31_regular(h21_s_out)

        h32_in = torch.cat((h21_s_out, h31_r_out), dim=1)

        class_mapping = self.classifier_l2(h32_in)

        return class_mapping


class Layer1Scale(nn.Module):

    def __init__(self, in_ch, out_ch, stride, pad):
        super().__init__()

        self.layer_1_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride, padding=pad)
        self.layer_1_bn = nn.BatchNorm2d(out_ch)
        self.layer_1_relu = nn.ReLU()

    def forward(self, x):
        x = self.layer_1_conv(x)
        x = self.layer_1_bn(x)
        x = self.layer_1_relu(x)

        return x


class LayerNScale(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride):
        super().__init__()

        self.layer_n_conv1 = nn.Conv2d(in_channels=in_ch, out_channels=mid_ch, kernel_size=1)
        self.layer_n_bn1 = nn.BatchNorm2d(mid_ch)
        self.layer_n_relu1 = nn.ReLU()
        self.layer_n_conv2 = nn.Conv2d(in_channels=mid_ch, out_channels=out_ch, kernel_size=3, stride=stride)
        self.layer_n_bn2 = nn.BatchNorm2d(out_ch)
        self.layer_n_relu2 = nn.ReLU()

    def forward(self, x):

        x = self.layer_n_conv1(x)
        x = self.layer_n_bn1(x)
        x = self.layer_n_relu1(x)
        x = self.layer_n_conv2(x)
        x = self.layer_n_bn2(x)
        x = self.layer_n_relu2(x)

        return x


class Classifier(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()

        self.classifier_conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3)
        self.classifier_conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3)
        self.classifier_avgpool = nn.AvgPool2d(3)

        self.dense = nn.Linear(out_ch * 3 * 3, NUM_CLASSES)

    def forward(self, x):
        x = self.classifier_conv1(x)
        x = self.classifier_conv2(x)
        x = self.classifier_avgpool(x)

        x = x.view(x.shape[0], -1)
        x = self.dense(x)

        return x
