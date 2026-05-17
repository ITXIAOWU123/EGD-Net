from model.Encoder_pvt import Encoder
import torch.nn as nn
from .decoder_m4 import Decoder
from model.lib.DuAT import *
from model.bgnet import *

import numpy as np
# import math
# from .tools import DeformableConv, image2patches

def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input

def get_sobel(in_chan, out_chan):
    '''
    filter_x = np.array([
        [3, 0, -3],
        [10, 0, -10],
        [3, 0, -3],
    ]).astype(np.float32)
    filter_y = np.array([
        [3, 10, 3],
        [0, 0, 0],
        [-3, -10, -3],
    ]).astype(np.float32)
    '''
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)
    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y






class Net(nn.Module):
    def __init__(self, opt,dim=128 , dims= [64, 128, 320, 512]):
        super(Net, self).__init__()

        self.encoder = Encoder()

        self.encoder_shaper_8 = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 512), nn.GELU())
        self.encoder_shaper_16 = nn.Sequential(nn.LayerNorm(320), nn.Linear(320, 320), nn.GELU())
        self.encoder_shaper_32 = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 128), nn.GELU())
        self.encoder_shaper_64 = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 64), nn.GELU())

        self.p = 8
        self.p2 = 16
        self.p3 = 32
        self.p4 = 64


        self.sobel_x1, self.sobel_y1 = get_sobel(64, 1)

        self.sobel_x4, self.sobel_y4 = get_sobel(512, 1)

        self.BEM = EAM()


        self.efm1 = EFM(64)
        self.efm2 = EFM(128)
        self.efm3 = EFM(320)
        self.efm4 = EFM(512)

        # self.cam1 = CAM(128, 64)
        # self.cam2 = CAM(256, 128)
        # self.cam3 = CAM(256, 256)

        # channels = 128
        # c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]

        self.decoder = Decoder(dim)
        




    def forward(self, x):
        B = x.shape[0]
        # PVT encoder
        out_8r, out_16r, out_32r, out_64r = self.encoder(x)
        pred = list()

        c4 = self.encoder_shaper_8(out_8r).transpose(1, 2).reshape(B, 512, self.p, self.p)
        c3 = self.encoder_shaper_16(out_16r).transpose(1, 2).reshape(B, 320, self.p * 2, self.p * 2)
        c2 = self.encoder_shaper_32(out_32r).transpose(1, 2).reshape(B, 128, self.p * 4, self.p * 4)
        c1 = self.encoder_shaper_64(out_64r).transpose(1, 2).reshape(B, 64, self.p * 8, self.p * 8)

        s1 = run_sobel(self.sobel_x1, self.sobel_y1, c1)
        s4 = run_sobel(self.sobel_x4, self.sobel_y4, c4)
        edge = self.BEM(s4, s1)

        edge_att = torch.sigmoid(edge)



        viz_data = {
            'input': x.detach().cpu(),
            'c4': c4.detach().cpu(),
            'c1': c1.detach().cpu(),
            'raw_edge': edge.detach().cpu(),  # SBA原始输出
            'edge_att': torch.sigmoid(edge).detach().cpu()  # 激活后的边缘图
        }
        
        # patches_batch1 = image2patches(x, patch_ref=c1, transformation="b c (hg h) (wg w) -> b (c hg wg) h w")
        # patches_batch2 = image2patches(x, patch_ref=c2, transformation="b c (hg h) (wg w) -> b (c hg wg) h w")
        # patches_batch3 = image2patches(x, patch_ref=c3, transformation="b c (hg h) (wg w) -> b (c hg wg) h w")
        # patches_batch4 = image2patches(x, patch_ref=c4, transformation="b c (hg h) (wg w) -> b (c hg wg) h w")
        
        x1a = self.efm1(c1, edge_att)
        x2a = self.efm2(c2, edge_att)
        x3a = self.efm3(c3, edge_att)
        x4a = self.efm4(c4, edge_att)
        shape = (256, 256)
        #Decoder
        P5, P4, P3, P2, P1,D41,D31,D21,D11 = self.decoder(x4a, x3a, x2a, x1a, shape)
        edge_o = F.interpolate(edge, size=shape, mode='bilinear', align_corners=False)
        pred.append(edge_o)
        pred.append(P5)
        pred.append(P4)
        pred.append(P3)
        pred.append(P2)
        pred.append(P1)







        viz_data1 = {
            # 输入
            'input': x.detach().cpu(),

            # Encoder 输出
            'c1': c1.detach().cpu(),
            'c2': c2.detach().cpu(),
            'c3': c3.detach().cpu(),
            'c4': c4.detach().cpu(),

            'raw_edge': edge.detach().cpu(),
            'edge_att': torch.sigmoid(edge).detach().cpu(),


            'efm_x1_after': x1a.detach().cpu(),
            'efm_x2_after': x2a.detach().cpu(),
            'efm_x3_after': x3a.detach().cpu(),
            'efm_x4_after': x4a.detach().cpu(),

            # CAM 前后
            'cam4_after': D41.detach().cpu(),
            'cam3_after': D31.detach().cpu(),
            'cam2_after': D21.detach().cpu(),
            'cam1_after': D11.detach().cpu(),

            # 最终预测
            'pred_5': P5.detach().cpu(),
            'pred_4': P4.detach().cpu(),
            'pred_3': P3.detach().cpu(),
            'pred_2': P2.detach().cpu(),
            'pred_1': P1.detach().cpu(),
        }

        

        return pred, viz_data, viz_data1
