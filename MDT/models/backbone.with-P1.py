#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch.nn as nn
import torch.nn.functional as F
import torch


class FPN(nn.Module):
    """
    Feature Pyramid Network from https://arxiv.org/pdf/1612.03144.pdf with options for modifications.
    by default is constructed with Pyramid levels P2, P3, P4, P5.
    """
    def __init__(self, cf, conv, operate_stride1=False):
        """
        from configs:
        :param input_channels: number of channel dimensions in input data.
        :param start_filts:  number of feature_maps in first layer. rest is scaled accordingly.
        :param out_channels: number of feature_maps for output_layers of all levels in decoder.
        :param conv: instance of custom conv class containing the dimension info.
        :param res_architecture: string deciding whether to use "resnet50" or "resnet101".
        :param operate_stride1: boolean flag. enables adding of Pyramid levels P1 (output stride 2) and P0 (output stride 1).
        :param norm: string specifying type of feature map normalization. If None, no normalization is applied.
        :param relu: string specifying type of nonlinearity. If None, no nonlinearity is applied.
        :param sixth_pooling: boolean flag. enables adding of Pyramid level P6.
        """
        super(FPN, self).__init__()

        self.start_filts = cf.start_filts
        start_filts = self.start_filts
        self.n_blocks = [3, 4, {"resnet50": 6, "resnet101": 23}[cf.res_architecture], 3]
        self.block = ResBlock
        self.block_expansion = 4
        self.operate_stride1 = operate_stride1
        self.sixth_pooling = cf.sixth_pooling
        self.dim = conv.dim

        if operate_stride1:
            self.C0 = nn.Sequential(conv(cf.n_channels, start_filts, ks=3, pad=1, norm=cf.norm, relu=cf.relu),
                                    conv(start_filts, start_filts, ks=3, pad=1, norm=cf.norm, relu=cf.relu))

            self.C1 = conv(start_filts, start_filts, ks=7, stride=(2, 2, 1) if conv.dim == 3 else 2, pad=3, norm=cf.norm, relu=cf.relu)

        else:
            # GG New P1 output
            # self.C1 = conv(cf.n_channels, start_filts, ks=7, stride=(2, 2, 1) if conv.dim == 3 else 2, pad=3, norm=cf.norm, relu=cf.relu)
            self.C1 = conv(cf.n_channels, start_filts*2, ks=3, stride=(2, 2, 1) if conv.dim == 3 else 2, pad=1, norm=cf.norm, relu=cf.relu)
        start_filts_exp = start_filts * self.block_expansion

        C2_layers = []
        # GG M1 C2_layers.append(nn.MaxPool2d( kernel_size=3, stride=2, padding=1)
        #                 if conv.dim == 2 else nn.MaxPool3d(kernel_size=3, stride=(2, 2, 1), padding=1))
        # c_in = start_filts
        # GG M1 C2_layers.append( self.block( c_in, c_in*4, conv=conv, stride=1, norm=cf.norm, relu=cf.relu, downsample=True) )
        c_in = start_filts *2
        C2_layers.append( self.block( c_in, c_in*2, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=True))
        c_in = start_filts*4
        for i in range(1, self.n_blocks[0]):
            C2_layers.append(self.block(c_in, c_in, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C2 = nn.Sequential(*C2_layers)

        C3_layers = []
        C3_layers.append(self.block( c_in, c_in*2, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=True))
        c_in = start_filts*8
        for i in range(1, self.n_blocks[1]):
            C3_layers.append(self.block( c_in, c_in, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C3 = nn.Sequential(*C3_layers)

        C4_layers = []
        C4_layers.append( self.block( c_in, c_in*2, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=True))
        c_in = start_filts*16
        for i in range(1, self.n_blocks[2]):
            C4_layers.append(self.block( c_in, c_in, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C4 = nn.Sequential(*C4_layers)

        C5_layers = []
        C5_layers.append(self.block( c_in, c_in*2, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=True))
        c_in = start_filts*32
        for i in range(1, self.n_blocks[3]):
            C5_layers.append( self.block( c_in, c_in, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C5 = nn.Sequential(*C5_layers)

        if self.sixth_pooling:
            C6_layers = []
            C6_layers.append( self.block( c_in, c_in*2, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=True ))
            c_in = start_filts*64
            for i in range(1, self.n_blocks[3]):
                C6_layers.append( self.block( c_in, c_in, conv=conv, norm=cf.norm, relu=cf.relu))
            self.C6 = nn.Sequential(*C6_layers)

        if conv.dim == 2:
            self.P1_upsample = Interpolate(scale_factor=2, mode='bilinear')
            self.P2_upsample = Interpolate(scale_factor=2, mode='bilinear')
        else:
            self.P1_upsample = Interpolate(scale_factor=(2, 2, 1), mode='trilinear')
            self.P2_upsample = Interpolate(scale_factor=(2, 2, 1), mode='trilinear')

        # Build Mk (convolution part)
        self.out_channels = cf.end_filts
        self.P5_conv1 = conv(start_filts*32 + cf.n_latent_dims, self.out_channels, ks=1, stride=1, relu=None) #
        self.P4_conv1 = conv(start_filts*16, self.out_channels, ks=1, stride=1, relu=None)
        self.P3_conv1 = conv(start_filts*8, self.out_channels, ks=1, stride=1, relu=None)
        self.P2_conv1 = conv(start_filts*4, self.out_channels, ks=1, stride=1, relu=None)
        # GG P1 self.P1_conv1 = conv(start_filts, self.out_channels, ks=1, stride=1, relu=None)
        self.P1_conv1 = conv(start_filts*2, self.out_channels, ks=1, stride=1, relu=None)

        if operate_stride1:
            self.P0_conv1 = conv(start_filts, self.out_channels, ks=1, stride=1, relu=None)
            self.P0_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)

        # GG Mk -> Pk
        self.P1_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P2_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P3_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P4_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P5_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)

        if self.sixth_pooling:
            self.P6_conv1 = conv(start_filts * 64, self.out_channels, ks=1, stride=1, relu=None)
            self.P6_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)


    def forward(self, x):
        """
        :param x: input image of shape (b, c, y, x, (z))
        :return: list of output feature maps per pyramid level, each with shape (b, c, y, x, (z)).
        """
        if self.operate_stride1:
            c0_out = self.C0(x)
        else:
            c0_out = x

        c1_out = self.C1(c0_out)
        c2_out = self.C2(c1_out)
        c3_out = self.C3(c2_out)
        c4_out = self.C4(c3_out)
        c5_out = self.C5(c4_out)
        if self.sixth_pooling:
            print ("ATTENTION: self.sixth_pooling")
            c6_out = self.C6(c5_out)
            p6_pre_out = self.P6_conv1(c6_out)
            p5_pre_out = self.P5_conv1(c5_out) + F.interpolate(p6_pre_out, scale_factor=2)
        else:
            p5_pre_out = self.P5_conv1(c5_out)

        p4_pre_out = self.P4_conv1(c4_out) + F.interpolate(p5_pre_out, scale_factor=2)
        p3_pre_out = self.P3_conv1(c3_out) + F.interpolate(p4_pre_out, scale_factor=2)
        p2_pre_out = self.P2_conv1(c2_out) + F.interpolate(p3_pre_out, scale_factor=2)
        # GG P1
        p1_pre_out = self.P1_conv1(c1_out) + F.interpolate(p2_pre_out, scale_factor=2)

        # plot feature map shapes for debugging.
        # for ii in [c0_out, c1_out, c2_out, c3_out, c4_out, c5_out]:
        #     print ("encoder shapes:", ii.shape)

        # GG P1 p1_out added
        p1_out = self.P1_conv2(p1_pre_out)
        p2_out = self.P2_conv2(p2_pre_out)
        p3_out = self.P3_conv2(p3_pre_out)
        p4_out = self.P4_conv2(p4_pre_out)
        p5_out = self.P5_conv2(p5_pre_out)
        # GG P1
        # out_list = [p2_out, p3_out, p4_out, p5_out]
        out_list = [p1_out, p2_out, p3_out, p4_out, p5_out]


        if self.sixth_pooling:
            p6_out = self.P6_conv2(p6_pre_out)
            out_list.append(p6_out)

        if self.operate_stride1:
            p1_pre_out = self.P1_conv1(c1_out) + self.P2_upsample(p2_pre_out)
            p0_pre_out = self.P0_conv1(c0_out) + self.P1_upsample(p1_pre_out)
            # p1_out = self.P1_conv2(p1_pre_out) # usually not needed.
            p0_out = self.P0_conv2(p0_pre_out)
            out_list = [p0_out] + out_list
        #
        # for i, ii in enumerate(out_list):
        #     print("decoder shapes:", i, ii.shape)

        return out_list



class ResBlock(nn.Module):

    def __init__(self, c_in, c_out, conv, stride=1, norm=None, relu='relu', downsample=False):
        super(ResBlock, self).__init__()
        # c_out must be divided by 4
        if c_out % 4 != 0:
            print('Error, not allowed: c_out must be divided by 4 ')
            exit()

        c_interm = c_out // 4
        # print( type(c_in), type(c_interm), type(c_out), type(stride), type(norm), type(relu), type(downsample) )
        # print( c_in, c_interm, c_out, stride, norm, relu, downsample )

        self.conv1 = conv(c_in, c_interm, ks=1, stride=stride, norm=norm, relu=relu)
        self.conv2 = conv(c_interm, c_interm, ks=3, pad=1, norm=norm, relu=relu)
        self.conv3 = conv(c_interm, c_out, ks=1, norm=norm, relu=None)
        self.relu = nn.ReLU(inplace=True) if relu == 'relu' else nn.LeakyReLU(inplace=True)
        if downsample :
          self.downsample = conv(c_in, c_out, ks=1, stride=stride, norm=norm, relu=None)            
        else:
          self.downsample = None
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x
