from MSD.settings.config import NUM_CLASSES, SCALE_CHANNELS, IMAGE_COLOUR_MODE

import torch
from torch import nn


class MSDnet(nn.Module):

    def __init__(self):
        super().__init__()
        in_size = 1 if IMAGE_COLOUR_MODE == 'L' else 3
        self.initial_reduce = nn.Sequential(
            nn.Conv2d(in_channels=in_size, out_channels=3, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # layer 1
        self.h11_regular = Layer1Scale(in_ch=3,
                                       out_ch=SCALE_CHANNELS['scale_1'], stride=1, pad=1)

        self.h21_stride = Layer1Scale(in_ch=SCALE_CHANNELS['scale_1'],
                                      out_ch=SCALE_CHANNELS['scale_2'], stride=2, pad=2)

        self.h31_stride = Layer1Scale(in_ch=SCALE_CHANNELS['scale_2'],
                                      out_ch=SCALE_CHANNELS['scale_3'], stride=2, pad=2)

        # layer 2
        self.h12_regular = LayerNScale(in_ch=SCALE_CHANNELS['scale_1'],
                                       mid_ch=SCALE_CHANNELS['scale_1'],
                                       out_ch=SCALE_CHANNELS['scale_1'], stride=1, pad=1)

        self.h22_stride = LayerNScale(in_ch=SCALE_CHANNELS['scale_1'],
                                      mid_ch=SCALE_CHANNELS['scale_2'],
                                      out_ch=SCALE_CHANNELS['scale_2'], stride=2, pad=2)
        self.h22_regular = LayerNScale(in_ch=SCALE_CHANNELS['scale_2'],
                                       mid_ch=SCALE_CHANNELS['scale_2'],
                                       out_ch=SCALE_CHANNELS['scale_2'], stride=1, pad=1)

        self.h32_stride = LayerNScale(in_ch=SCALE_CHANNELS['scale_2'],
                                      mid_ch=SCALE_CHANNELS['scale_3'],
                                      out_ch=SCALE_CHANNELS['scale_3'], stride=2, pad=2)
        self.h32_regular = LayerNScale(in_ch=SCALE_CHANNELS['scale_3'],
                                       mid_ch=SCALE_CHANNELS['scale_3'],
                                       out_ch=SCALE_CHANNELS['scale_3'], stride=1, pad=1)
        # layer 3
        self.h23_stride = LayerNScale(in_ch=SCALE_CHANNELS['scale_1']*2,
                                      mid_ch=SCALE_CHANNELS['scale_2'],
                                      out_ch=SCALE_CHANNELS['scale_2'], stride=2, pad=2)
        self.h23_regular = LayerNScale(in_ch=SCALE_CHANNELS['scale_2']*2,
                                      mid_ch=SCALE_CHANNELS['scale_2'],
                                      out_ch=SCALE_CHANNELS['scale_2'], stride=1, pad=1)

        self.h33_stride = LayerNScale(in_ch=SCALE_CHANNELS['scale_2']*2,
                                      mid_ch=SCALE_CHANNELS['scale_3'],
                                      out_ch=SCALE_CHANNELS['scale_3'], stride=2, pad=2)
        self.h33_regular = LayerNScale(in_ch=SCALE_CHANNELS['scale_3']*2,
                                      mid_ch=SCALE_CHANNELS['scale_3'],
                                      out_ch=SCALE_CHANNELS['scale_3'], stride=1, pad=1)

        # layer 4
        self.h34_stride = LayerNScale(in_ch=SCALE_CHANNELS['scale_2']*3,
                                      mid_ch=SCALE_CHANNELS['scale_3'],
                                      out_ch=SCALE_CHANNELS['scale_3'], stride=2, pad=2)

        self.h34_regular = LayerNScale(in_ch=SCALE_CHANNELS['scale_3']*3,
                                      mid_ch=SCALE_CHANNELS['scale_3'],
                                      out_ch=SCALE_CHANNELS['scale_3'], stride=1, pad=1)


        # classifiers
        self.classifier_l2 = Classifier(in_ch=SCALE_CHANNELS['scale_3']*2,
                                        mid_ch=128,
                                        out_ch=128,
                                        in_shape=3)

        self.classifier_l3 = Classifier(in_ch=SCALE_CHANNELS['scale_3']*4,
                                        mid_ch=128,
                                        out_ch=128, in_shape=3)

        self.classifier_l4 = Classifier(in_ch=SCALE_CHANNELS['scale_3']*6,
                                        mid_ch=128,
                                        out_ch=128, in_shape=3)


    def forward(self, x):
        x_reduced = self.initial_reduce(x)

        # layer 1
        x_1_1 = self.h11_regular(x_reduced)
        print(f"x11 shape: {x_1_1.shape}")

        x_2_1 = self.h21_stride(x_1_1)
        print(f"x21 shape: {x_2_1.shape}")

        x_3_1 = self.h31_stride(x_2_1)
        #print(f"x31 shape: {x_1_1.shape}")

        # layer 2
        x_1_2 = self.h12_regular(x_1_1)

        print(f"x22 stride shape: {self.h22_stride(x_1_1).shape}")
        print(f"x22 regular shape: {self.h22_stride(x_1_1).shape}")

        quit()

        x_2_2 = torch.cat((
            self.h22_stride(x_1_1),
            self.h22_regular(x_2_1)
            ), dim=1
        )

        x_3_2 = torch.cat((
            self.h32_stride(x_2_1),
            self.h32_regular(x_3_1)
            ), dim=1
        )

        # layer 3
        x_2_3 = torch.cat((
            self.h23_stride(torch.cat((x_1_1, x_1_2), dim=1)),
            self.h23_regular(torch.cat((x_2_1, x_2_2), dim=1))
            ), dim=1
        )

        x_3_3 = torch.cat((
            self.h33_stride(torch.cat((x_2_1, x_2_2), dim=1)),
            self.h33_regular(torch.cat((x_3_1, x_3_2), dim=1))
            ), dim=1
        )

        # layer 4
        x_3_4 = torch.cat((
            self.h34_stride(torch.cat((x_2_1, x_2_2, x_2_3), dim=1)),
            self.h34_regular(torch.cat((x_3_1, x_3_2, x_3_3), dim=1))
            ), dim=1
        )

        # classifiers

        classifier_layer_2 = self.classifier_l2(x_3_2)
        classifier_layer_3 = self.classifier_l3(x_3_3)
        classifier_layer_4 = self.classifier_l4(x_3_4)

        return classifier_layer_2, classifier_layer_3, classifier_layer_4


class Layer1Scale(nn.Module):

    def __init__(self, in_ch, out_ch, stride, pad):
        super().__init__()

        self.layer_1_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                                      kernel_size=3, stride=stride, padding=pad, bias=False)
        self.layer_1_bn = nn.BatchNorm2d(out_ch)
        self.layer_1_relu = nn.ReLU()

    def forward(self, x):
        x = self.layer_1_conv(x)
        x = self.layer_1_bn(x)
        x = self.layer_1_relu(x)

        return x


class LayerNScale(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride, pad):
        super().__init__()

        self.layer_n_conv1 = nn.Conv2d(in_channels=in_ch, out_channels=mid_ch,
                                       kernel_size=1, padding=0, bias=False)
        self.layer_n_bn1 = nn.BatchNorm2d(mid_ch)
        self.layer_n_relu1 = nn.ReLU()
        self.layer_n_conv2 = nn.Conv2d(in_channels=mid_ch, out_channels=out_ch,
                                       kernel_size=3, stride=stride, padding=1, bias=False)
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
    def __init__(self, in_ch, mid_ch, out_ch, in_shape):
        super().__init__()

        self.classifier_conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=0)
        self.classifier_conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=0)
        self.classifier_avgpool = nn.AvgPool2d(3)

        self.dense = nn.Linear(out_ch * in_shape**2,
                               NUM_CLASSES)

    def forward(self, x):
        x = self.classifier_conv1(x)
        x = self.classifier_conv2(x)
        x = self.classifier_avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)

        return x
