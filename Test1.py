# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 17:41:30 2021
@author: angelou
"""

import torch
from torch.autograd import Variable
import os
import argparse
import cv2
from datetime import datetime
from utils.dataloader import get_loader,test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
from torchstat import stat
from collections import OrderedDict
from MFTN import MFTN



def structure_loss(pred, mask):
    
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)

    pt = torch.exp(-wbce)
    F_loss = (1-pt)**2 * wbce

    return (wbce + wiou).mean() + torch.mean(F_loss)

def JAC(dec_score):
    JAC_score = (dec_score + 1) / (2 - dec_score + 1)
    return JAC_score
def test(model, path):
    
    ##### put ur data_path of TestDataSet/Kvasir here #####
    data_path = path
    #####                                             #####
    
    model.eval()
    image_root = '{}/image/'.format(data_path)
    gt_root = '{}/label/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, 352)
    b=0.0
    print('[test_size]',test_loader.size)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        # print(image.shape)
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        lateral_map_5= model(image, test='O')
        res = lateral_map_5
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res[torch.where(res>0)] /= (res>0).float().mean()
        res[torch.where(res<0)] /= (res<0).float().mean()
        res  = res.sigmoid().data.cpu().numpy().squeeze()
        
        # input = res
        input = (res > 0.5).astype(int)
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input,(-1))
        target_flat = np.reshape(target,(-1))
 
        intersection = (input_flat*target_flat)
        
        # loss =  (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)

        # loss = (intersection.sum() + smooth) / (input.sum() + target.sum() - intersection.sum() + smooth)

        # tp = ((input_flat == 1) & (target_flat == 1)).sum().item()
        # fp = ((input_flat == 1) & (target_flat == 0)).sum().item()
        # pre = (tp + smooth) / (tp + fp + smooth)

        # loss = ((input_flat == target_flat).sum().item()) / (((target_flat == 0) | (target_flat == 1)).sum().item())

        # tn = np.sum((input_flat == 0)& (target_flat == 0))
        # fp = np.sum((input_flat == 1) & (target_flat == 1))
        # loss = (tn + smooth) / (tn + fp + smooth)

        tp = np.sum((input_flat == 1)& (target_flat == 1))
        fn = np.sum((input_flat == 0) & (target_flat == 1))
        recall = (tp + smooth) / (tp + fn + smooth)

        # loss = 2 * pre * recall / (pre + recall)

        # predicted_image = input_flat.flatten()
        # true_label = target_flat.flatten()
        # # 计算平方差
        # squared_diff = (predicted_image - true_label) ** 2
        # # 计算平均均方误差
        # loss = np.mean(squared_diff)

        # from scipy.stats import spearmanr
        # num_samples = input.shape[0]
        # loss = 0.0

        # for i in range(num_samples):
        #     pred_i = input[i].ravel()
        #     target_i = target[i].ravel()

        #     pred_rank = np.argsort(pred_i)
        #     target_rank = np.argsort(target_i)

        #     loss += spearmanr(pred_rank, target_rank).correlation

        # loss /= num_samples
        # print(loss)
        # fp = open('log/loss.txt','a')
        # fp.write(str(loss) +" " + name +'\n')
        # fp.close()
        # cv2.imwrite('log/pic/'+name, np.round(res*255))
        # print('log/pic/'+name)
        a =  '{:.4f}'.format(recall)
        a = float(a)
        b = b + a
        
    return b/3309

#保存


            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epoch', type=int,
                        default=200, help='epoch number')
    
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    
    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='choosing optimizer Adam or SGD')
    
    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')
    
    parser.add_argument('--batchsize', type=int,
                        default=6, help='training batch size')
    
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    
    parser.add_argument('--train_path', type=str,
                        default='/home/server/sda2/wth_data/BraTS2018/2-MICCAI_BraTS_2018/0504try/train', help='path to train dataset')
    
    parser.add_argument('--test_path', type=str,
                        default='/home/server/sda2/wth_data/BraTS2018/2-MICCAI_BraTS_2018/0504try/test' , help='path to testing Kvasir dataset')
    
    parser.add_argument('--train_save', type=str,
                        default='MFTN-best')
    
    parser.add_argument('--testsize', type=int, default=224, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./snapshots/MFTN-best/MFTN-best.pth')
    
    opt = parser.parse_args()

    # ---- build models ----
    torch.cuda.set_device(0)  # set your gpu device
    model = MFTN().cuda()


    params = model.parameters()
    
    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, opt.lr)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay = 1e-4, momentum = 0.9)
        
    weights = torch.load(opt.pth_path)
    new_state_dict = OrderedDict()

    for k, v in weights.items():

    
        if 'total_ops' not in k and 'total_params' not in k:
            name = k
            new_state_dict[name] = v

        
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()
    meandice = test(model,opt.test_path)
    print(meandice)