# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 17:41:30 2021
@author: angelou
"""

import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from utils.dataloader import get_loader,test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
from torchstat import stat
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
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        
        res5= model(image, test='O')
        res = res5
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res[torch.where(res>0)] /= (res>0).float().mean()
        res[torch.where(res<0)] /= (res<0).float().mean()
        res  = res.sigmoid().data.cpu().numpy().squeeze()
        
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input,(-1))
        target_flat = np.reshape(target,(-1))
 
        intersection = (input_flat*target_flat)
        
        loss =  (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)

        a =  '{:.4f}'.format(loss)
        a = float(a)
        b = b + a
        
    return b/3309

#保存
def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    epochsij = {}
    epochsij['S'] = int(2)
    epochsij['I'] = int(6) 
    epochsij['J'] = int(8)
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    coarse_loss_record1, fine_loss_record2 = AvgMeter(), AvgMeter()
    for mode in ['S', 'I', 'J']:
        for e in range(epochsij[mode]):
            for i, pack in enumerate(train_loader, start=1):
                for rate in size_rates:
                    optimizer.zero_grad()
                    # ---- data prepare ----
                    images, gts = pack
                    # print(images.shape, gts.shape)
                    images = Variable(images).cuda()
                    gts = Variable(gts).cuda()
                    # ---- rescale ----
                    trainsize = int(round(opt.trainsize*rate/32)*32)
                    if rate != 1:
                        images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                        gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    # ---- forward ----
                    coarse_prob, fine_prob = model(images, gts, mode)
                    print(coarse_prob.shape, fine_prob.shape)
                    COARSE_WEIGHT = 1 / 3
                    # ---- loss function ----

                    coarse_loss = structure_loss(coarse_prob, gts)
                    fine_loss = structure_loss(fine_prob, gts)
                    loss = COARSE_WEIGHT * coarse_loss + (1 - COARSE_WEIGHT) * fine_loss
                    
                    # loss = loss5 +loss3 + loss2 + loss1
                    # ---- backward ----
                    loss.backward()
                    clip_gradient(optimizer, opt.clip)
                    optimizer.step()
                    # ---- recording loss ----
                    if rate == 1:
                        
                        coarse_loss_record1.update(coarse_loss.data, opt.batchsize)
                        fine_loss_record2.update(fine_loss.data, opt.batchsize)
                        # print(coarse_loss_record1, fine_loss_record2)
                # ---- train visualization ----
                if i % 20 == 0 or i == total_step:
                    print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                            ' lateral-5: {:0.4f}], lateral-3: {:0.4f}]]'.
                            format(datetime.now(), epoch, opt.epoch, i, total_step,
                                    coarse_loss_record1.show(),fine_loss_record2.show()))
            save_path = 'snapshots/{}/'.format(opt.train_save)
            os.makedirs(save_path, exist_ok=True)





            if (epoch+1) % 1 == 0:
                meandice = test(model,test_path)
                
                fp = open('log/log.txt','a')
                fp.write(str(meandice)+'\n')
                fp.close()
                
                fp = open('log/best.txt','r')
                best = fp.read()
                fp.close()
                
                if meandice > float(best):
                    fp = open('log/best.txt','w')
                    fp.write(str(meandice))
                    fp.close()
                    # best = meandice
                    fp = open('log/best.txt','r')
                    best = fp.read()
                    fp.close()
                    torch.save(model.state_dict(), save_path + 'MFTN-best.pth' )
                    print('[Saving Snapshot:]', save_path + 'MFTN-best.pth',meandice,'[best:]',best)

            

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
    
    opt = parser.parse_args()

    # ---- build models ----
    torch.cuda.set_device(0)  # set your gpu device
    model = MFTN().cuda()
    # ---- flops and params ----
    # from utils.utils import CalParams
    # x = torch.randn(1, 3, 352, 352).cuda()
    # CalParams(model, x)

    params = model.parameters()
    
    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, opt.lr)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay = 1e-4, momentum = 0.9)
        
    print(optimizer)
    image_root = '{}/image/'.format(opt.train_path)
    gt_root = '{}/label/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, augmentation = opt.augmentation)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt.test_path)