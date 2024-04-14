# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:58:14 2021

@author: angelou
"""

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pretrain.Res2Net_v1b import res2net50_v1b_26w_4s, res2net101_v1b_26w_4s
from res2net import Res2Net50, weight_init
import math
import torchvision.models as models
from lib.conv_layer import Conv, BNPReLU
from lib.axial_atten import AA_kernel
from lib.context_module import CFPModule
from lib.partial_decoder import aggregation
import os
 
    

    
class MFTN(nn.Module):
    def __init__(self, channel=32):
        super().__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.margin = 20
        # Receptive Field Block
        self.rfb2_1 = Conv(512, 32,3,1,padding=1,bn_acti=True)
        self.rfb3_1 = Conv(1024, 32,3,1,padding=1,bn_acti=True)
        self.rfb4_1 = Conv(2048, 32,3,1,padding=1,bn_acti=True)

        # Partial Decoder
        self.agg1 = aggregation(channel)
        
        self.CFP_1 = CFPModule(32, d = 8)
        self.CFP_2 = CFPModule(32, d = 8)
        self.CFP_3 = CFPModule(32, d = 8)
        ###### dilation rate 4, 62.8



        self.ra1_conv1 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra2_conv1 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra3_conv1 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.aa_kernel_1 = AA_kernel(32,32)
        self.aa_kernel_2 = AA_kernel(32,32)
        self.aa_kernel_3 = AA_kernel(32,32)
        
		# Saliency Transformation Module
        self.saliency1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.relu_saliency1 = nn.ReLU(inplace=True)
        self.saliency2 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2)

        # Fine-scaled Network
        self.fine_model = Res2Net50()
        self.linear5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.linear4 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.linear3 = nn.Sequential(nn.Conv2d( 512, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.predict = nn.Conv2d(64 * 3, 3, kernel_size=1, stride=1, padding=0)

        self.conv5 = nn.Conv2d(3, 1, 1) #这里修改

    def forward(self, x, label=None, mode=None,test=None):
        if test is None:
            image = x
            self.test = test
            image = self.resnet.conv1(image)
            image = self.resnet.bn1(image)
            image = self.resnet.relu(image)
            image = self.resnet.maxpool(image)      # bs, 64, 88, 88
            
            # ----------- low-level features -------------
            
            x1 = self.resnet.layer1(image)      # bs, 256, 88, 88
            x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
            
            x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
            x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11
            
            x2_rfb = self.rfb2_1(x2) # 512 - 32
            x3_rfb = self.rfb3_1(x3) # 1024 - 32
            x4_rfb = self.rfb4_1(x4) # 2048 - 32
            
            decoder_1 = self.agg1(x4_rfb, x3_rfb, x2_rfb) # 1,44,44
            lateral_map_1 = F.interpolate(decoder_1, scale_factor=8, mode='bilinear')
            # print(lateral_map_1.size(), x.size())
            lateral_map_1 = torch.sigmoid(lateral_map_1)
            coarse_prob = lateral_map_1
            # Saliency Transformation Module
            lateral_map_1 = self.relu_saliency1(self.saliency1(lateral_map_1))
            lateral_map_1 = self.saliency2(lateral_map_1)
            saliency = lateral_map_1
            # print(label.size(), image.size(), saliency.size())
            if mode == 'S':
                cropped_image, crop_info = self.crop(label, x)
            elif mode == 'I':
                cropped_image, crop_info = self.crop(label, x * saliency)
            elif mode == 'J':
                cropped_image, crop_info = self.crop(coarse_prob, x * saliency, label)
            else:
                raise ValueError("wrong value of mode, should be in ['S', 'I', 'J']")
            
            h = cropped_image
            out2, out3, out4, out5 = self.fine_model(h)
            # print(222222222222)
            # print(cropped_image.size(), out3.size(), out4.size(), out5.size())
            out5 = self.linear5(out5)
            out4 = self.linear4(out4)
            out3 = self.linear3(out3)
            out5 = F.interpolate(out5, size=out3.size()[2:], mode='bilinear', align_corners=True)
            out4 = F.interpolate(out4, size=out3.size()[2:], mode='bilinear', align_corners=True)
            pred = torch.cat([out5, out4*out5, out3*out4*out5], dim=1)
            # print(pred.size())
            h = self.predict(pred)
            h = F.interpolate(h, size=cropped_image.size()[2:], mode='bilinear', align_corners=True)
            # print(h.size())
            # print(111111111)
            h = self.uncrop(crop_info, h, x)
            h = torch.sigmoid(h)
            fine_prob = h
            coarse_prob = self.conv5(coarse_prob)
            fine_prob = self.conv5(fine_prob)
            # print(coarse_prob.size(), h.size())
            return coarse_prob, fine_prob
        elif test == 'O': # Oracle testing
            # assert label is not None and mode is None
            # Coarse-scaled Network
            h = x
            h = self.resnet.conv1(h)
            # print(x.size())
            h = self.resnet.bn1(h)
            # print(x.size())
            h = self.resnet.relu(h)
            # print(x.size())
            h = self.resnet.maxpool(h)      # bs, 64, 88, 88
            
            # ----------- low-level features -------------
            
            x1 = self.resnet.layer1(h)      # bs, 256, 88, 88
            x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
            
            x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
            x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11
            
            x2_rfb = self.rfb2_1(x2) # 512 - 32
            x3_rfb = self.rfb3_1(x3) # 1024 - 32
            x4_rfb = self.rfb4_1(x4) # 2048 - 32
            
            decoder_1 = self.agg1(x4_rfb, x3_rfb, x2_rfb) # 1,44,44
            # print(x2_rfb.size(), x3_rfb.size(), x4_rfb.size(), decoder_1.size())
            lateral_map_1 = F.interpolate(decoder_1, scale_factor=8, mode='bilinear')
            lateral_map_1 = torch.sigmoid(lateral_map_1)
            coarse_prob = lateral_map_1
            # coarse_prob = self.conv5(coarse_prob)
            # Saliency Transformation Module
            lateral_map_1 = self.relu_saliency1(self.saliency1(lateral_map_1))
            lateral_map_1 = self.saliency2(lateral_map_1)
            saliency = lateral_map_1
            label = np.zeros(x.shape, dtype = np.float32)
            cropped_image, crop_info = self.crop(coarse_prob, x * saliency)
            # Fine-scaled Network
            h = cropped_image
            out2, out3, out4, out5 = self.fine_model(h)
            # print(222222222222)
            # print(cropped_image.size(), out3.size(), out4.size(), out5.size())
            out5 = self.linear5(out5)
            out4 = self.linear4(out4)
            out3 = self.linear3(out3)
            out5 = F.interpolate(out5, size=out3.size()[2:], mode='bilinear', align_corners=True)
            out4 = F.interpolate(out4, size=out3.size()[2:], mode='bilinear', align_corners=True)
            pred = torch.cat([out5, out4*out5, out3*out4*out5], dim=1)
            # print(pred.size())
            h = self.predict(pred)
            h = F.interpolate(h, size=cropped_image.size()[2:], mode='bilinear', align_corners=True)
            h = self.uncrop(crop_info, h, x)
            evidence = F.softplus(h)
            h = torch.sigmoid(h)
            fine_prob = h
            fine_prob = self.conv5(fine_prob)
            return fine_prob, evidence
        else:
            raise ValueError("wrong value of TEST, should be in [None, 'C', 'F', 'O']")

    
    
    def crop(self, prob_map, saliency_data, label=None):
        (N, C, W, H) = prob_map.shape
        binary_mask = (prob_map >= 0.5) # torch.uint8
        if label is not None and binary_mask.sum().item() == 0:
            binary_mask = (label >= 0.5)

        self.left = self.margin
        self.right = self.margin
        self.top = self.margin
        self.bottom = self.margin
        if binary_mask.sum().item() == 0: # avoid this by pre-condition in TEST 'F'
            minA = 0
            maxA = W
            minB = 0
            maxB = H
            self.no_forward = True
        else:
            if N > 1:
                mask = torch.zeros(size = (N, C, W, H))
                for n in range(N):
                    cur_mask = binary_mask[n, :, :, :]
                    arr = torch.nonzero(cur_mask)

                    if arr[:, 1].numel() > 0:
                        minA = arr[:, 1].min().item()
                        maxA = arr[:, 1].max().item()
                    else:
                        minA = 0
                        maxA = W
                    if arr[:, 2].numel() > 0:
                        minB = arr[:, 2].min().item()
                        maxB = arr[:, 2].max().item()
                    else:
                        minB = 0
                        maxB = H
                    bbox = [int(max(minA - self.left, 0)), int(min(maxA + self.right + 1, W)), \
            int(max(minB - self.top, 0)), int(min(maxB + self.bottom + 1, H))]
                    mask[n, :, bbox[0]: bbox[1], bbox[2]: bbox[3]] = 1
                # print(saliency_data.size(), mask.size())
                saliency_data = saliency_data * mask.cuda()

            arr = torch.nonzero(binary_mask)

            if arr[:, 2].numel() > 0:
                minA = arr[:, 2].min().item()
                maxA = arr[:, 2].max().item()
            else:
                minA = 0
                maxA = W
            if arr[:, 3].numel() > 0:
                minB = arr[:, 3].min().item()
                maxB = arr[:, 3].max().item()
            else:
                minB = 0
                maxB = H
            self.no_forward = False

        bbox = [int(max(minA - self.left, 0)), int(min(maxA + self.right + 1, W)), \
            int(max(minB - self.top, 0)), int(min(maxB + self.bottom + 1, H))]
        cropped_image = saliency_data[:, :, bbox[0]: bbox[1], \
            bbox[2]: bbox[3]]

        # if self.no_forward == True and self.TEST == 'F':
        if self.no_forward == True:
            cropped_image = torch.zeros_like(cropped_image).cuda()

        crop_info = np.zeros((1, 4), dtype = np.int16)
        crop_info[0] = bbox
        crop_info = torch.from_numpy(crop_info).cuda()

        return cropped_image, crop_info



    def uncrop(self, crop_info, cropped_image, image):
        uncropped_image = torch.ones_like(image).cuda()
        uncropped_image *= (-9999999)
        bbox = crop_info[0]
        uncropped_image[:, :, bbox[0].item(): bbox[1].item(), bbox[2].item(): bbox[3].item()] = cropped_image
        return uncropped_image

    def update_margin(self):
        MAX_INT = 256
        if random.randint(0, MAX_INT - 1) >= MAX_INT * self.prob:
            self.left = self.margin
            self.right = self.margin
            self.top = self.margin
            self.bottom = self.margin
        else:
            a = np.zeros(self.batch * 4, dtype = np.uint8)
            for i in range(self.batch * 4):
                a[i] = random.randint(0, self.margin * 2)
            self.left = int(a[0: self.batch].sum() / self.batch)
            self.right = int(a[self.batch: self.batch * 2].sum() / self.batch)
            self.top = int(a[self.batch * 2: self.batch * 3].sum() / self.batch)
            self.bottom = int(a[self.batch * 3: self.batch * 4].sum() / self.batch)

if __name__ == '__main__':
    ras = MFTN().cuda()
    input_tensor = torch.randn(1, 3, 224, 224).cuda()

    out = ras(input_tensor)