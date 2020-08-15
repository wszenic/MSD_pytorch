from MSD.settings.config import NUM_CLASSES, SCALE_CHANNELS

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
        self.h11_stride = Layer1Scale(in_ch=3,
                                      out_ch=SCALE_CHANNELS['scale_1'], stride=2, pad=1)
        self.h21_stride = Layer1Scale(in_ch=SCALE_CHANNELS['scale_1'],
                                      out_ch=SCALE_CHANNELS['scale_2'], stride=2, pad=1)

        self.h11_regular = Layer1Scale(in_ch=3,
                                       out_ch=SCALE_CHANNELS['scale_1'], stride=1, pad=1)
        self.h21_regular = Layer1Scale(in_ch=SCALE_CHANNELS['scale_1'],
                                       out_ch=SCALE_CHANNELS['scale_2'], stride=1, pad=1)
        self.h31_regular = Layer1Scale(in_ch=SCALE_CHANNELS['scale_2'],
                                       out_ch=SCALE_CHANNELS['scale_3'], stride=1, pad=1)

        # layer 2
        self.h12_stride = LayerNScale(in_ch=SCALE_CHANNELS['scale_1'],
                                      mid_ch=SCALE_CHANNELS['scale_1'],
                                      out_ch=SCALE_CHANNELS['scale_1'], stride=2, pad=1)
        self.h22_stride = LayerNScale(in_ch=SCALE_CHANNELS['scale_1'] + SCALE_CHANNELS['scale_2'],
                                      mid_ch=SCALE_CHANNELS['scale_2'],
                                      out_ch=SCALE_CHANNELS['scale_2'], stride=2, pad=2)

        # self.h12_regular = LayerNScale(in_ch=SCALE_CHANNELS['scale_1'],
        #                                mid_ch=SCALE_CHANNELS['scale_1'],
        #                                out_ch=SCALE_CHANNELS['scale_1'], stride=0, pad=1)
        self.h22_regular = LayerNScale(in_ch=SCALE_CHANNELS['scale_1'] + SCALE_CHANNELS['scale_2'],
                                       mid_ch=SCALE_CHANNELS['scale_2'],
                                       out_ch=SCALE_CHANNELS['scale_2'], stride=1, pad=1)
        self.h32_regular = LayerNScale(in_ch=SCALE_CHANNELS['scale_2'] + SCALE_CHANNELS['scale_3'],
                                       mid_ch=SCALE_CHANNELS['scale_3'],
                                       out_ch=SCALE_CHANNELS['scale_3'], stride=1, pad=1)


        # layer 3
        self.h23_stride = LayerNScale(in_ch=SCALE_CHANNELS['scale_2'] * 2,
                                      mid_ch=SCALE_CHANNELS['scale_2'],
                                      out_ch=SCALE_CHANNELS['scale_2'], stride=2, pad=2)

        self.h33_regular = LayerNScale(in_ch=SCALE_CHANNELS['scale_3'] * 2,
                                      mid_ch=SCALE_CHANNELS['scale_3'],
                                      out_ch=SCALE_CHANNELS['scale_3'], stride=1, pad=2)


        # classifiers
        self.classifier_l2 = Classifier(in_ch=SCALE_CHANNELS['scale_3'] + SCALE_CHANNELS['scale_2'],
                                        mid_ch=SCALE_CHANNELS['scale_3'] + SCALE_CHANNELS['scale_2'],
                                        out_ch=SCALE_CHANNELS['scale_3'] + SCALE_CHANNELS['scale_2'],
                                        in_shape=3)

        self.classifier_l3 = Classifier(in_ch= SCALE_CHANNELS['scale_3']*2,
                                        mid_ch= SCALE_CHANNELS['scale_3']*2,
                                        out_ch= SCALE_CHANNELS['scale_3']*2, in_shape=3)

        self.classifier_l4 = Classifier(in_ch= SCALE_CHANNELS['scale_3']*3,
                                        mid_ch= SCALE_CHANNELS['scale_3']*3,
                                        out_ch= SCALE_CHANNELS['scale_3']*3, in_shape=3)

    def forward(self, x):
        x_reduced = self.initial_reduce(x)
        # x_reduced = x
        # print(f'x_reduced shape: {x_reduced.shape}')
        # layer 1
        h11_s_out = self.h11_stride(x_reduced)
        # print(f'h11_s_out shape: {h11_s_out.shape}')

        h11_r_out = self.h11_regular(x_reduced)
        # print(f'h11_r_out shape: {h11_r_out.shape}')
        h21_s_out = self.h21_stride(h11_s_out)
        # print(f'h21_s_out shape: {h21_s_out.shape}')

        h21_r_out = self.h21_regular(h11_s_out)
        # print(f'h21_r_out shape: {h21_r_out.shape}')
        h31_r_out = self.h31_regular(h21_s_out)
        # print(f'h31_r_out shape: {h31_r_out.shape}')

        # print('-'*20)
        # layer 2
        h12_s_out = self.h12_stride(h11_r_out)
        # print(f'h12_s_out shape: {h12_s_out.shape}')

        h22_in = torch.cat((h11_s_out, h21_r_out), dim=1)
        # print(f'h22_in shape: {h22_in.shape}')
        h22_s_out = self.h22_stride(h22_in)
        # print(f'h22_s_out shape: {h22_s_out.shape}')
        h22_r_out = self.h22_regular(h22_in)
        # print(f'h22_r_out shape: {h22_r_out.shape}')

        h32_in = torch.cat((h21_s_out, h31_r_out), dim=1)
        # print(f'h32_in shape: {h32_in.shape}')
        h32_r_out = self.h32_regular(h32_in)
        # print(f'h32_r_out shape: {h32_r_out.shape}')

        # layer 3
        # print('-'*20)
        h23_in = torch.cat((h21_r_out, h22_r_out), dim=1)
        # print(f'h23_in shape: {h23_in.shape}')
        h23_s_out = self.h23_stride(h23_in)
        # print(f'h23_s_out shape: {h23_s_out.shape}')

        h33_in = torch.cat((h31_r_out, h32_r_out), dim=1)
        # print(f'h33_in shape: {h33_in.shape}')
        h33_r_out = self.h33_regular(h33_in)
        # print(f'h33_r_out shape: {h33_r_out.shape}')

        #layer 4 <- virtual, as only serves as classifier input
        h41_in = torch.cat((h33_in, h33_r_out), dim=1)

        #classifiers
        classifier_layer_2 = self.classifier_l2(h32_in)

        classifier_layer_3 = self.classifier_l3(h33_in)

        classifier_layer_4 = self.classifier_l4(h41_in)

        return classifier_layer_2, classifier_layer_3, classifier_layer_4


class Layer1Scale(nn.Module):

    def __init__(self, in_ch, out_ch, stride, pad):
        super().__init__()

        self.layer_1_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                                      kernel_size=3, stride=stride, padding=pad)
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
                                       kernel_size=1, padding=0)
        self.layer_n_bn1 = nn.BatchNorm2d(mid_ch)
        self.layer_n_relu1 = nn.ReLU()
        self.layer_n_conv2 = nn.Conv2d(in_channels=mid_ch, out_channels=out_ch,
                                       kernel_size=3, stride=stride, padding=1)
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

        self.classifier_conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3)
        self.classifier_conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3)
        self.classifier_avgpool = nn.AvgPool2d(3)

        self.dense = nn.Linear(out_ch * in_shape**2,
                               NUM_CLASSES)

    def forward(self, x):
        x = self.classifier_conv1(x)
        x = self.classifier_conv2(x)
        x = self.classifier_avgpool(x)
        # print(f'x shape: {x.shape}')
        x = x.view(x.shape[0], -1)
        # print(f'x shape: {x.shape}')
        x = self.dense(x)
        # print(f'x shape: {x.shape}')

        return x
