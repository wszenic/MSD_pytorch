from MSD.settings.config import NUM_CLASSES, SCALE_CHANNELS,\
    CLASSIFIER_SCALES, IMAGE_COLOUR_MODE

import torch
from torch import nn


class MSDnet(nn.Module):

    def __init__(self):
        super().__init__()

        self.colour_channels = 3 if IMAGE_COLOUR_MODE == 'RGB' else 1

        self.initial_reduce = nn.Sequential(
            nn.Conv2d(in_channels=self.colour_channels, out_channels=3, kernel_size=7, stride=2),
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
                                      out_ch=SCALE_CHANNELS['scale_1'], stride=2)
        self.h22_stride = LayerNScale(in_ch=SCALE_CHANNELS['scale_1'] + SCALE_CHANNELS['scale_2'],
                                      mid_ch=SCALE_CHANNELS['scale_2'],
                                      out_ch=SCALE_CHANNELS['scale_2'], stride=2)

        # self.h12_regular = LayerNScale(in_ch=SCALE_CHANNELS['scale_1'],
        #                                mid_ch=SCALE_CHANNELS['scale_1'],
        #                                out_ch=SCALE_CHANNELS['scale_1'], stride=0, pad=1)
        self.h22_regular = LayerNScale(in_ch=SCALE_CHANNELS['scale_1'] + SCALE_CHANNELS['scale_2'],
                                       mid_ch=SCALE_CHANNELS['scale_2'],
                                       out_ch=SCALE_CHANNELS['scale_2'], stride=1)
        self.h32_regular = LayerNScale(in_ch=SCALE_CHANNELS['scale_2'] + SCALE_CHANNELS['scale_3'],
                                       mid_ch=SCALE_CHANNELS['scale_3'],
                                       out_ch=SCALE_CHANNELS['scale_3'], stride=1)


        # layer 3
        self.h23_stride = LayerNScale(in_ch=SCALE_CHANNELS['scale_2'] * 2 + SCALE_CHANNELS['scale_1'] * 2,
                                      mid_ch=SCALE_CHANNELS['scale_2'],
                                      out_ch=SCALE_CHANNELS['scale_2'], stride=2)

        self.h33_regular = LayerNScale(in_ch=SCALE_CHANNELS['scale_3'] * 2 + SCALE_CHANNELS['scale_2'] * 2,
                                      mid_ch=SCALE_CHANNELS['scale_3'],
                                      out_ch=SCALE_CHANNELS['scale_3'], stride=1)


        # classifiers
        self.classifier_l2 = Classifier(in_ch=SCALE_CHANNELS['scale_3'] + SCALE_CHANNELS['scale_2'],
                                        mid_ch=(SCALE_CHANNELS['scale_3'] + SCALE_CHANNELS['scale_2']) // 2,
                                        out_ch=CLASSIFIER_SCALES['out_ch'],
                                        in_shape=3)

        self.classifier_l3 = Classifier(in_ch=SCALE_CHANNELS['scale_3']*2 + SCALE_CHANNELS['scale_2'] * 2,
                                        mid_ch=(SCALE_CHANNELS['scale_3']*2 + SCALE_CHANNELS['scale_2'] * 2) // 2,
                                        out_ch=CLASSIFIER_SCALES['out_ch'], in_shape=3)

        self.classifier_l4 = Classifier(in_ch=SCALE_CHANNELS['scale_3']*3 + SCALE_CHANNELS['scale_2'] * 5,
                                        mid_ch=(SCALE_CHANNELS['scale_3']*3 + SCALE_CHANNELS['scale_2'] * 5) // 2,
                                        out_ch=CLASSIFIER_SCALES['out_ch'], in_shape=3)

    def forward(self, x):
        x_reduced = self.initial_reduce(x)
        # layer 1
        h11_s_out = self.h11_stride(x_reduced)

        h11_r_out = self.h11_regular(x_reduced)
        h21_s_out = self.h21_stride(h11_s_out)

        h21_r_out = self.h21_regular(h11_s_out)
        h31_r_out = self.h31_regular(h21_s_out)

        # layer 2
        h12_s_out = self.h12_stride(h11_r_out)

        h22_in = torch.cat((h11_s_out, h21_r_out), dim=1)
        h22_s_out = self.h22_stride(h22_in)
        h22_r_out = self.h22_regular(h22_in)

        h32_in = torch.cat((h21_s_out, h31_r_out), dim=1)
        h32_r_out = self.h32_regular(h32_in)

        # layer 3
        h23_in = torch.cat((h21_r_out, h22_r_out,
                            h11_s_out, h12_s_out), dim=1)
        h23_s_out = self.h23_stride(h23_in)

        h33_in = torch.cat((h31_r_out, h32_r_out,
                            h21_s_out, h22_s_out), dim=1)
        h33_r_out = self.h33_regular(h33_in)

        #layer 4 <- virtual, as only serves as classifier input
        h43_in = torch.cat((h33_in, h33_r_out,
                            h21_s_out, h22_s_out, h23_s_out), dim=1)

        #classifiers
        classifier_layer_2 = self.classifier_l2(h32_in)

        classifier_layer_3 = self.classifier_l3(h33_in)

        classifier_layer_4 = self.classifier_l4(h43_in)

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
    def __init__(self, in_ch, mid_ch, out_ch, stride):
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
        x = x.view(x.shape[0], -1)
        x = self.dense(x)

        return x
