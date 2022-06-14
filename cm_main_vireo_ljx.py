# -*- coding: utf-8 -*-
import io
import os
import os.path
import time
import random
import argparse
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torchvision import transforms
from cm_build_dataset_ljx import build_dataset
from cm_utils import *
import itertools
#load environmental settings
import cm_opts_ljx
opt = cm_opts_ljx.opt_algorithm()
# if opt.stage == 1:
#     from vilt_model_ljx.model import ViLTransformerSS,VilT_Classification
# elif opt.stage == 2:
#     from vilt_model_ljx.model_word import ViLTransformerSS,VilT_Classification
# elif opt.stage == 3 :
#     if opt.method == 'IT':
#         from vilt_model_ljx.model_full import ViLTransformerSS,VilT_Classification
#     elif opt.method == 'T2I':
#         from vilt_model_ljx.model_full_t2i import ViLTransformerSS,VilT_Classification
#     else:
#         from vilt_model_ljx.model_full import ViLTransformerSS,VilT_Classification
from vilt_model_ljx.model_pha import ViLTransformerSS,VilT_Classification

# import torch
# torch.backends.cudnn.enabled = False

#-----------------------------------------------------------------dataset information--------------------------------------------------------------------

opt.dataset = 'vireo'

opt.root_path = '/vireo/' # path to root folder
opt.img_path = '/vireo172/ready_chinese_food'# path to image folder
opt.data_path = opt.root_path + 'SplitAndIngreLabel/' # path to data folder
opt.food_class_name_path = opt.data_path + 'FoodList.txt'# path to the list of names for classes
if opt.method in ['recon','img2word']:
    opt.dataset_num_class = 353 # number of classes in the dataset
else:
    opt.dataset_num_class = 172 # number of classes in the dataset
opt.dataset_max_seq = 15 # max number of word for a sample in train data
opt.dataset_max_seq_test = 11 # max number of word for a sample in test data
opt.dataset_num_word = 353 # number of ingredients in the dataset

image_size = [384,384]

if opt.img_net == 'resnet18':
    path_img_net_pretrain = '/mnt/model_pretrain/datset=vireo~stage=1~bs=64~lr=0.0001~lrd=4~wd=0.001~lrd_rate=0.1/model_best.pt'
elif opt.img_net == 'resnet50':
    path_img_net_pretrain = '/mnt/model_pretrain//datset=vireo~stage=1~img_net=resnet50~bs=64~lr=5e-05~lrd=4~wd=0.001~lrd_rate=0.1/model_best.pt'
else:
    path_img_net_pretrain = ''
path_word_net_pretrain = '/mnt/model_pretrain/datset=vireo~stage=2~bs=64~lr=0.001~lrd=4~wd=0.001~lrd_rate=0.1/model_best.pt'
path_word_decoder_pretrain = '/mnt/model_pretrain/datset=vireo~stage=2~word_net=gru~method=recon~bs=64~lr=0.005~lrd=4~wd=0.001~lrd_rate=0.1~lr_finetune=0.0005~lrd_finetune=0.1~update_word_encoder/model_best.pt'


#--------------------------------------------------------------------settings----------------------------------------------------------------------------

# basic
CUDA = 1  # 1 for True; 0 for False
SEED = 1
measure_best = 0 # best measurement
torch.manual_seed(SEED)
kwargs = {'num_workers': 5, 'pin_memory': True} if CUDA else {}
if CUDA:
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)
    
# log and model paths
if opt.exp == 4:
    opt.result_path = '/mnt/results/exp4/'

# ipdb.set_trace()
result_path = os.path.join(opt.result_path, para_name(opt))

if not os.path.exists(result_path):
        os.makedirs(result_path)

#train settings
mode = opt.mode
latent_len = 300
dim_align = 128
EPOCHS = opt.lr_decay * 3 + 1
if opt.test_only == True:
    EPOCHS = 1

#-------------------------------------------------------------dataset & dataloader-----------------------------------------------------------------------
class RandomColorDropping:
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if self.p > random.random():
            idx = random.randint(0, 2)
            new_img = img.split()[idx]
            return new_img.convert('RGB')
        return img
    
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_img_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size)
])
transform_img_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size)
])
# 仅原图生成图对比时使用
transform_img_ori = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size)
])

#contrastive learning
def train_transform(input_size=384, crop=True, use_blur=True):
    '''
    :param input_size: the size of image
    :param crop: true to set RandomResizedCrop
    :param use_blur: true to use GaussianBlur
    :return: transforms
    '''
    t = [transforms.RandomHorizontalFlip(p=0.5)]
    t.append(transforms.Resize(input_size))
    if crop:
        t.append(transforms.RandomResizedCrop(size=input_size))
    t.append(transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8))
    t.append(transforms.RandomGrayscale(p=0.2))
#     t.append(RandomColorDropping(p=0.5))
    if use_blur:
        t.append(transforms.GaussianBlur(kernel_size=9))
    t.append(transforms.ToTensor())
#     t.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]))
#     t.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))
    return transforms.Compose(t)
class ContrastiveTransformations:
    def __init__(self, n_views=2, input_size=384, use_blur=True, crop=True, transform=None, transform_ori=transform_img_ori):
        self.input_size = input_size
        if transform is None:
            self.base_transforms = train_transform(self.input_size, crop=crop, use_blur=use_blur)
        else:
            self.base_transforms = transform
        self.n_views = n_views
        self.transform_ori = transform_ori

    def __call__(self, x):
        return [self.transform_ori(x),self.base_transforms(x)]#生成原图
#         return [self.base_transforms(x) for _ in range(self.n_views)]

if opt.contrastive == True and opt.method in ['I','IL','T2I']:
    transform_img_train = ContrastiveTransformations()
    transform_img_test = ContrastiveTransformations()


    
#create dataset
dataset_train = build_dataset(opt.stage, opt.img_path, opt.data_path, transform_img_train, mode, opt.dataset, 'train')
dataset_test = build_dataset(opt.stage, opt.img_path, opt.data_path, transform_img_test, mode, opt.dataset, 'test')
#dataloader
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, **kwargs)
# wrn/vgg -> batch_size may not 100
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=200, shuffle=False, **kwargs)

class RandomMask(nn.Module):
    def __init__(self, p1, p2):
        super(RandomMask, self).__init__()
        self.p1 = p1
        self.p2 = p2

    def forward(self, x):
#         ipdb.set_trace()
        if torch.rand(1) < self.p1:
            r = torch.rand(size=x.shape[:2], device=x.device)
            mask = r > self.p2
#             mask = mask.unsqueeze(2)
            x = x * mask.float()
        return x
randmask = RandomMask(0.1,0.1)

#-----------------------------------------------------------------Model/optimizer----------------------------------------------------------------------------------
    
def get_updateModel(model, path):
#     ipdb.set_trace()
    pretrained_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()
 
    shared_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(shared_dict)
#    load nocls
#     model_dict['module2.dense1.weight']=pretrained_dict['module2.dense1.weight'][:,-144:]#后144
#     model_dict['module2.dense1.bias']=pretrained_dict['module2.dense1.bias']
#    IT T2I
#     model_dict['module2.dense1.weight']=pretrained_dict['module2.dense1.weight'][:,-144:]#15+1+144
#     model_dict['module2.dense1.bias']=pretrained_dict['module2.dense1.bias']
#     model_dict['module2.dense1_t.weight']=pretrained_dict['module2.dense1.weight'][:,:15]
#     model_dict['module2.dense1_t.bias']=pretrained_dict['module2.dense1.bias']
    
#     model_dict['module2.dense1.weight']=pretrained_dict['module2.dense1_i.weight']
#     model_dict['module2.dense1.bias']=pretrained_dict['module2.dense1_i.bias']

    
    model.load_state_dict(model_dict)

    return model

# model define
# from cm_mymodel import build_mymodel
# net_type = [opt.img_net, opt.word_net]
# pretrain_path = [path_img_net_pretrain, path_word_net_pretrain, path_word_decoder_pretrain]

from vilt_model_ljx.config import get_config
vilt_config = get_config()
model = VilT_Classification(vilt_config)

if opt.exp == 4:
    model = get_updateModel(model,vilt_config['load_path'])
model.cuda()


# divided

# params_module1 = [v for k, v in model.named_parameters() if not
#                 (k.startswith('module2.dense2') or k.startswith('module2.dense1'))
#                ] 
# params_module2 = [v for k, v in model.named_parameters() if
#                 (k.startswith('module2.dense2') or k.startswith('module2.dense1'))
#                ] 
params_module1 = []
key_module1 = []
params_module2 = []
key_module2 = []
key_module_fr = []

FTblks = [i for i in range(opt.frozen_blks,12)]
for k,v in model.named_parameters():
    if k.startswith('module1.transformer.blocks'):
        if int(k.split('.')[3]) in FTblks:
            print(k)
            params_module1 += [v] #部分网络
            key_module1 += [k]
        else:
            key_module_fr += [k]
# for k, v in model.named_parameters():
    elif k.startswith('module1.transformer.norm'):
        params_module1 += [v]
        key_module1 += [k]
    elif k.startswith('module2'):
        params_module2 += [v]
        key_module2 += [k]
    else:
        key_module_fr += [k]
        v.requires_grad =False
          
params_m1 = [{'params': params_module1}]
params_m2 = [{'params': params_module2}]
optimizer_m1 = optim.Adam(params_m1,weight_decay=opt.weight_decay, lr=opt.lr_m1)
optimizer_m2 = optim.Adam(params_m2,weight_decay=opt.weight_decay, lr=opt.lr_m2)
optimizer = [optimizer_m1,optimizer_m2]

# for k, v in model.named_parameters():
# #     ipdb.set_trace()
#     if k in key_module_fr:
#         v.requires_grad =False


# frozen
# params_module2 = [v for k, v in model.named_parameters() if
#                 (k.startswith('module2.dense2'))
#                ] 
# params_module2 = [v for k, v in model.named_parameters() if
#                 (k.startswith('module2.dense2') or k.startswith('module2.dense1'))
#                ] 
# params_m2 = [{'params': params_module2}]
# optimizer = optim.Adam(params_m2, weight_decay=opt.weight_decay, lr=opt.lr_m2)
# no frozen
# optimizer = optim.Adam(model.parameters(), weight_decay=opt.weight_decay, lr=opt.lr_m1)

def set_classifier_optimizer(model, opt):
    # params in ingre prediction net
    pretrained_dict = torch.load(opt.path_pretrain_v, map_location='cpu')
    model_dict = model.state_dict()
    
    for k, v in model.named_parameters():
        if k in pretrained_dict:
            v.requires_grad =False
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=opt.weight_decay, lr=opt.lr)
    
    return optimizer
# ----------------------------------------------------------------contrstive loss----------------------------------------------------------------------------------
from torch import Tensor
import math
class NCELoss(torch.nn.Module):#加标签信息的对比学习loss
    def __init__(self, temperature=0.1):
        super(NCELoss, self).__init__()
        self.EPISILON = 1e-10
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=1)

    def forward(self, f1, f2, targets):
        ### cuda implementation
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
#         ipdb.set_trace()
        ## set distances of the same label to zeros
        mask = targets.unsqueeze(1) - targets
        self_mask = (torch.zeros_like(mask) != mask)  ### where the negative samples are labeled as 1
        dist = (f1.unsqueeze(1) - f2).pow(2).sum(2)
        ## convert l2 distance to cos distance
        cos = 1 - 0.5 * dist

        ## convert cos distance to exponential space
        pred_softmax = self.softmax(cos / self.temperature)  ### convert to multi-class prediction scores

        log_pos_softmax = - torch.log(pred_softmax + self.EPISILON) * ((~self_mask).float())
        log_neg_softmax = - torch.log(1 - pred_softmax + self.EPISILON) * (self_mask).float()
        log_softmax = log_pos_softmax.sum(1) / (~self_mask).sum(1).float() + log_neg_softmax.sum(1) / self_mask.sum(
            1).float()
        loss = log_softmax

        return loss.mean()
    
class NTXentLoss(nn.Module):#不加标签信息的loss
    def __init__(self, temperature=0.1, eps=1e-6):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, out_1, out_2):
#         ipdb.set_trace()
        out = torch.cat([out_1, out_2], dim=0)

        neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature).sum(dim=-1)

        row_sub = Tensor(neg.shape).fill_(math.e ** (1 / self.temperature)).to(neg.device)

        neg = torch.clamp(neg - row_sub, min=self.eps)

        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        return -torch.log(pos / (neg + self.eps)).mean()
# ----------------------------------------------------------------Train----------------------------------------------------------------------------------
def imshow(img):
    # img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.savefig('/mnt/results/img')


def train_epoch(epoch, decay, optimizer, train_stage, train_log):
    
    # vireo measurement
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    losses = AverageMeter('Loss', ':.4e')
    losses_con = AverageMeter('Loss', ':.4e')
    losses_con2 = AverageMeter('Loss', ':.4e')
    losses_cls1 = AverageMeter('Loss', ':.4e')
    losses_cls2 = AverageMeter('Loss', ':.4e')

    model.train()
    total_time = time.time()
#     ipdb.set_trace()
    for batch_idx, (data, label) in enumerate(train_loader):
           
#         ipdb.set_trace()        
        print("- - -- - - - - - -batch_idx : {}".format(batch_idx))
        start_time = time.time()
        # load data
        if train_stage == 1:
            if opt.contrastive == True :
                batch_size_cur = data[0].size(0)             
                if CUDA:
                    x1, x2 = data
                    x1 = x1.cuda()
                    x2 = x2.cuda()
                    label = label.cuda()
            else:
                batch_size_cur = data.size(0)
                if CUDA:
                    data = data.cuda()
                    label = label.cuda()
        elif train_stage == 2 or train_stage == 4:
#             ipdb.set_trace()
            [indexVectors, words] = data
            batch_size_cur = indexVectors.size(0)
            
            if CUDA:
                indexVectors = indexVectors.cuda()
                words = words.cuda()
                label = label.cuda()
        elif train_stage == 3:

            [img, indexVectors, words] = data
            batch_size_cur = indexVectors.size(0)
            if opt.VSaug == True:
                img1, img2 = img  #aug
                indexVectors1 = indexVectors  #目前的策略是原图+增广+原文+增广
                indexVectors2 = randmask(indexVectors)
    #             ipdb.set_trace()
                if CUDA:
                    indexVectors1 = indexVectors1.cuda()
                    indexVectors2 = indexVectors2.cuda()
                    img1 = img1.cuda()
                    img2 = img2.cuda()
                    words = words.cuda()
                    label = label.cuda()
            else:
                if CUDA:
                    indexVectors = indexVectors.cuda()
                    img = img.cuda()
                    words = words.cuda()
                    label = label.cuda()
        else:
            assert 1 < 0, 'Please fill train_stage!'

        # prediction and loss
        if train_stage == 1:
            if opt.contrastive == True :

                feats1, out1 = model(x1)
                feats2, out2 = model(x2)
                if opt.method == 'IL':
                    criterion_con = NCELoss()#temp 0.1 or 0.5
                    loss_con = criterion_con(feats1, feats2, label)
#                 ipdb.set_trace()
                elif opt.method == 'I':
                    criterion_con = NTXentLoss()#temp 0.1 or 0.5
                    loss_con = criterion_con(feats1, feats2)
#                 loss_con = criterion_con(out1, out2)
                criterion = nn.CrossEntropyLoss()
                loss_cls1 = criterion(out1, label)
                loss_cls2 = criterion(out2, label)
                final_loss = loss_con + loss_cls1 + loss_cls2
            else:
                #perform prediction
                output = model(data)           
                #compute loss
                criterion = nn.CrossEntropyLoss()
                loss_cls = criterion(output, label)                
                final_loss = loss_cls

        elif train_stage == 2:
            # chose word net
            if opt.word_net == 'gru':
                if opt.method == 'recon':
                    output, label = model(indexVectors)
                    batch_size_cur = label.shape[0]
                    if CUDA:
                        label = label.cuda()
                else:
                    output = model(indexVectors)
            elif opt.word_net == 'nn':
                if opt.method == 'recon':
                    output = model(words)
                    label = words
                    batch_size_cur = label.shape[0]
                    if CUDA:
                        label = label.cuda()
                else:
                    output = model(words)
                #compute loss
            else:
#                 assert 1 < 0, 'Please indicate the correct word_net_type!'
#                 output = model(words)
                output = model(indexVectors)
        
            criterion = nn.CrossEntropyLoss()
            loss_cls = criterion(output, label)
            # final_loss = loss_cls
            final_loss = loss_cls  
            
        elif train_stage == 3:
#full直接使用多模态数据分类
            if opt.method in ['IT','T2I']: 
#                 ipdb.set_trace()
                if opt.VSaug ==True: 
                    feats_i, feats_t, output, out_t = model(img1, indexVectors1)  #aug
                    feats_i2, feats_t2, output2, out_t2 = model(img2, indexVectors2)
                    criterion_con = NTXentLoss()
                    loss_con1 = criterion_con(feats_i, feats_t)
                    loss_con2 = criterion_con(feats_t, feats_i)
                    loss_vcon = criterion_con(feats_i, feats_i2)
                    loss_scon = criterion_con(feats_t, feats_t2)
                    criterion = nn.CrossEntropyLoss()
                    loss_cls1 = criterion(output, label)
                    loss_cls_v2 = criterion(output2, label)
                    loss_cls2 = criterion(out_t, label)
                    loss_cls_s2 = criterion(out_t2, label)
                    loss_novsaug =  0.1*(loss_con1 + loss_con2 ) +( loss_cls1 ) + (loss_cls2)  #多模态，但没有模态内增广
                    final_loss =  loss_novsaug + 0.05*(loss_vcon + loss_scon)
#                     final_loss = 0.05*(loss_vcon + loss_scon)+(loss_cls1+loss_cls2)
#                     final_loss = 0.05*(loss_vcon )+(loss_cls1)
                else:
                    feats_i, feats_t, output, out_t = model(img, indexVectors) #noaug
                    criterion_con = NTXentLoss()
                    loss_con1 = criterion_con(feats_i, feats_t)
                    loss_con2 = criterion_con(feats_t, feats_i)
                    criterion = nn.CrossEntropyLoss()
                    loss_cls1 = criterion(output, label)
                    loss_cls2 = criterion(out_t, label)
                    final_loss =  (loss_con1 + loss_con2 ) +( loss_cls1 ) + (loss_cls2)
            else:
#                 origin stage 3:feature fusion classifiction
#                 output = model(img, indexVectors, buimg = False)
#                 criterion = nn.CrossEntropyLoss()
#                 loss_cls = criterion(output, label)                
#                 final_loss = loss_cls
# #         PHA CICAI        
                image_embeds_ll,text_embeds_ll,image_embeds_dv,text_embeds_dv,image_feats,text_feats,output = model(img, indexVectors)
                criterion = nn.CrossEntropyLoss()
                loss_cls = criterion(output, label)           
                if opt.type_align == 'l2':
                    criterion_align = nn.MSELoss()
                    loss_align = criterion_align(image_feats,text_feats)
                elif opt.type_align == 'kl':
                    criterion_align = nn.KLDivLoss()
                    image_feats = F.log_softmax(image_feats, dim=1)
                    text_feats = F.softmax(text_feats.detach(), dim=1)
                    loss_align = criterion_align(image_feats,text_feats)
                else:
                    criterion_align_l2 = nn.MSELoss()
                    loss_align_l2 = criterion_align_l2(image_feats,text_feats)
                    
                    criterion_align_kl = nn.KLDivLoss()
                    image_feats = F.log_softmax(image_feats, dim=1)
                    text_feats = F.softmax(text_feats.detach(), dim=1)
                    loss_align_kl = criterion_align_kl(image_feats,text_feats)
                    loss_align = 0.01*loss_align_l2 + loss_align_kl
                final_loss = loss_cls + loss_align

            
        elif train_stage == 4:            
            output = model(indexVectors, label)
            if opt.dataset == 'vireo':
                criterion_cls = nn.CrossEntropyLoss()
            loss_cls = criterion_cls(output, label)
            final_loss = loss_cls
            
            
        # optimization
        losses.update(final_loss.item(), batch_size_cur)
        if opt.contrastive == True:
            if opt.method in ['I','IL']:
                losses_con.update(loss_con.item(), batch_size_cur)
                losses_cls1.update(loss_cls1.item(), batch_size_cur)
                losses_cls2.update(loss_cls2.item(), batch_size_cur)
            else:
                losses_con.update(loss_con1.item(), batch_size_cur)
                losses_con2.update(loss_con2.item(), batch_size_cur)
                losses_cls1.update(loss_cls1.item(), batch_size_cur)
                losses_cls2.update(loss_cls2.item(), batch_size_cur)
                

#frozen/no frozen
#         optimizer.zero_grad()
#         final_loss.backward()
#         optimizer.step()
# divided
        [optimizer_m1, optimizer_m2] = optimizer
        optimizer_m1.zero_grad()
        optimizer_m2.zero_grad()
        final_loss.backward()
        optimizer_m1.step()
        optimizer_m2.step()
        
        # upddate log
        if train_stage in [1,2,4]:
            if opt.contrastive == True:
                acc1_1, acc1_5 = accuracy(out1, label, topk=(1, 5))
                acc2_1, acc2_5 = accuracy(out2, label, topk=(1, 5))
                acc1 = [(i+j)/2 for i,j in zip(acc1_1, acc2_1)]
                acc5 = [(i+j)/2 for i,j in zip(acc1_5, acc2_5)]
            else:                
                acc1, acc5 = accuracy(output, label, topk=(1, 5))
            top1.update(acc1[0], batch_size_cur)
            top5.update(acc5[0], batch_size_cur)
            
        elif train_stage == 3:
#full多模态直接分类 
            acc1, acc5 = accuracy(output, label, topk=(1, 5))
            top1.update(acc1[0], batch_size_cur)
            top5.update(acc5[0], batch_size_cur)
            
#full 里面加了3，原本的3注释掉了        
        if train_stage in [1,2,3,4]:
            if opt.contrastive == True:
                
# frozen/no frozen
#                 if opt.method in ['recon','img2word']:
#                     optimizer_cur = optimizer_tune
#                 else:
#                     optimizer_cur = optimizer
#                 log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
#                           'Time {data_time:.3f}\t'
#                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                            'Loss_con {loss_con.val:.4f} ({loss_con.avg:.4f})\t'
#                            'Loss_cls1 {loss_cls1.val:.4f} ({loss_cls1.avg:.4f})\t'
#                            'Loss_cls2 {loss_cls2.val:.4f} ({loss_cls2.avg:.4f})\t'
#                           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                           'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                     epoch, batch_idx, len(train_loader), data_time=round((time.time() - total_time), 4), loss=losses,loss_con=losses_con,loss_cls1=losses_cls1,loss_cls2=losses_cls2, top1=top1, top5=top5, lr=optimizer_cur.param_groups[-1]['lr']))
    # divided
                log_out = ('Epoch: [{0}][{1}/{2}], lr_m1: {lr_m1:.5f}\t lr_m2: {lr_m2:.5f}\t'
                          'Time {data_time:.3f}\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                           'Loss_con {loss_con.val:.4f} ({loss_con.avg:.4f})\t'
                           'Loss_cls1 {loss_cls1.val:.4f} ({loss_cls1.avg:.4f})\t'
                           'Loss_cls2 {loss_cls2.val:.4f} ({loss_cls2.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, batch_idx, len(train_loader), data_time=round((time.time() - total_time), 4), loss=losses,loss_con=losses_con,loss_cls1=losses_cls1,loss_cls2=losses_cls2, top1=top1, top5=top5, lr_m1=optimizer_m1.param_groups[-1]['lr'],lr_m2=optimizer_m2.param_groups[-1]['lr']))
            else:
#                 if opt.method in ['recon','img2word']:
#                     optimizer_cur = optimizer_tune
#                 else:
#                     optimizer_cur = optimizer
#                 log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
#                           'Time {data_time:.3f}\t'
#                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                           'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                     epoch, batch_idx, len(train_loader), data_time=round((time.time() - total_time), 4), loss=losses,top1=top1, top5=top5, lr=optimizer_cur.param_groups[-1]['lr']))
    # divided
                log_out = ('Epoch: [{0}][{1}/{2}], lr_m1: {lr_m1:.5f}\t lr_m2: {lr_m2:.5f}\t'
                          'Time {data_time:.3f}\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, batch_idx, len(train_loader), data_time=round((time.time() - total_time), 4), loss=losses, top1=top1, top5=top5, lr_m1=optimizer_m1.param_groups[-1]['lr'],lr_m2=optimizer_m2.param_groups[-1]['lr']))
                

            # print(output)
            train_log.write(log_out + '\n')
            train_log.flush()
        

# ----------------------------------------------------------------Test----------------------------------------------------------------------------------
def test_epoch(epoch, stage, test_log):
    # vireo measurement
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1_word = AverageMeter('Acc@1', ':6.2f')
    top5_word = AverageMeter('Acc@5', ':6.2f')
    losses = AverageMeter('Loss', ':.4e')
    losses_con = AverageMeter('Loss', ':.4e')
    losses_con2 = AverageMeter('Loss', ':.4e')
    losses_cls1 = AverageMeter('Loss', ':.4e')
    losses_cls2 = AverageMeter('Loss', ':.4e')
    model.eval()
    start_time = time.time()
    class_correct = [0. for _ in range(172)]
    total_correct = [0. for _ in range(172)]
    class_acc = {}
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
#             ipdb.set_trace()
            print("- - -- - - - - - -test_batch_idx : {}".format(batch_idx))
            # load data
            if stage == 1:                
                if opt.contrastive == True:
                    batch_size_cur = data[0].size(0)             
                    if CUDA:
                        x1, x2 = data
                        x1 = x1.cuda()
                        x2 = x2.cuda()
                        label = label.cuda()
                    feats1, out1 = model(x1)
                    feats2, out2 = model(x2)
                    if opt.method == 'IL':
                        criterion_con = NCELoss()#temp 0.1 or 0.5
                        loss_con = criterion_con(feats1, feats2, label)
    #                 ipdb.set_trace()
                    elif opt.method == 'I':
                        criterion_con = NTXentLoss()#temp 0.1 or 0.5
                        loss_con = criterion_con(feats1, feats2)

                    criterion = nn.CrossEntropyLoss()
                    loss_cls1 = criterion(out1, label)
                    loss_cls2 = criterion(out2, label)                    
                    final_loss = loss_con + loss_cls1 +loss_cls2
                    losses.update(final_loss.item(), batch_size_cur)
                    losses_con.update(loss_con.item(), batch_size_cur)
                    losses_cls1.update(loss_cls1.item(), batch_size_cur)
                    losses_cls2.update(loss_cls2.item(), batch_size_cur)

                    acc1_1, acc1_5 = accuracy(out1, label, topk=(1, 5))
                    acc2_1, acc2_5 = accuracy(out2, label, topk=(1, 5))
                    acc1 = [(i+j)/2 for i,j in zip(acc1_1, acc2_1)]
                    acc5 = [(i+j)/2 for i,j in zip(acc1_5, acc2_5)]
                    top1.update(acc1[0], batch_size_cur)
                    top5.update(acc5[0], batch_size_cur)
                    
                else:
                    batch_size_cur = data.size(0)
                    if CUDA:
                        data = data.cuda()
                        label = label.cuda()
                    output = model(data)
                    #compute loss
                    criterion = nn.CrossEntropyLoss()
                    loss_cls = criterion(output, label)
                    acc1, acc5 = accuracy(output, label, topk=(1, 5))
                    top1.update(acc1[0], batch_size_cur)
                    top5.update(acc5[0], batch_size_cur)
                    losses.update(loss_cls.item(), batch_size_cur)
                    
            elif stage == 2:
                [indexVectors, words] = data
                batch_size_cur = indexVectors.size(0)
                if CUDA:
                    indexVectors = indexVectors.cuda()
                    words = words.cuda()
                    label = label.cuda()
                if opt.word_net == 'gru':
                    if opt.method == 'recon':
                        output, label = model(indexVectors)
                        batch_size_cur = label.shape[0]
                        if CUDA:
                            label = label.cuda()
                    else:
                        output = model(indexVectors)
                elif opt.word_net == 'nn':
                    
                    if opt.method == 'recon':
                        output = model(words)
                        label = words
                        batch_size_cur = label.shape[0]
                        if CUDA:
                            label = label.cuda()
                    else:
                        output = model(words)
                else:#for vilt
                    output = model(indexVectors)
                #compute loss
                criterion = nn.CrossEntropyLoss()
                loss_cls = criterion(output, label)
                acc1, acc5 = accuracy(output, label, topk=(1, 5))
                top1.update(acc1[0], batch_size_cur)
                top5.update(acc5[0], batch_size_cur)
                losses.update(loss_cls.item(), batch_size_cur)
                    
                    
            elif stage == 3:
#                 # load data                
#                 feats_i, feats_t, output, out_t = model(img1, indexVectors1)
#                 feats_i2, feats_t2, output2, out_t2 = model(img2, indexVectors2)
#                 criterion_con = NTXentLoss()
#                 loss_con1 = criterion_con(feats_i, feats_t)
#                 loss_con2 = criterion_con(feats_t, feats_i)
#                 loss_vcon = criterion_con(feats_i, feats_i2)
#                 loss_scon = criterion_con(feats_t, feats_t2)
#                 criterion = nn.CrossEntropyLoss()
#                 loss_cls1 = criterion(output, label)
#                 loss_cls2 = criterion(out_t, label)
#                 final_loss = 0.1* (loss_con1 + loss_con2 +loss_vcon + loss_son) + loss_cls1 + 0.1* loss_cls2

                [img, indexVectors, words] = data
                batch_size_cur = indexVectors.size(0)
                if CUDA:
                    indexVectors = indexVectors.cuda()
                    if opt.VSaug == True:
                        img1, img2 = img #aug
                        img1 = img1.cuda()
                        words = words.cuda()
                        label = label.cuda()
                    else:
                        img = img.cuda()
                        words = words.cuda()
                        label = label.cuda()
                if opt.method in ['IT','T2I']: 
                    if opt.VSaug == True:
                        img =img1
#                         indexVectors = randmask(indexVectors)
                    feats_i, feats_t, output, out_t = model(img, indexVectors)
                    criterion_con = NTXentLoss()
                    loss_con1 = criterion_con(feats_i, feats_t)
                    loss_con2 = criterion_con(feats_t, feats_i)
                    criterion = nn.CrossEntropyLoss()
                    loss_cls1 = criterion(output, label)
                    loss_cls2 = criterion(out_t, label)
                    final_loss = loss_cls1
                    losses.update(final_loss.item(), batch_size_cur)
                    losses_con.update(loss_con1.item(), batch_size_cur)
                    losses_con2.update(loss_con2.item(), batch_size_cur)
                    losses_cls1.update(loss_cls1.item(), batch_size_cur)
                    losses_cls2.update(loss_cls2.item(), batch_size_cur)
                    acc1, acc5 = accuracy(output, label, topk=(1, 5))
                else:                   
#full 
                    image_embeds_ll,text_embeds_ll,image_embeds_dv,text_embeds_dv,image_feats,text_feats,output = model(img, indexVectors)
#                     image_embeds_pt = torch.cat([image_embeds_pt, image_embeds.detach().cpu()], dim=0)
#                     embeds_jh_pt = torch.cat([embeds_jh_pt, embeds_jh.detach().cpu()], dim=0)
#                     feats_pt = torch.cat([feats_pt, feats.detach().cpu()], dim=0)                 
#                     t_embeds_pt = torch.cat([t_embeds_pt, t_embeds.detach().cpu()], dim=0)
#                     t_embeds_jh_pt = torch.cat([t_embeds_jh_pt, t_embeds_jh.detach().cpu()], dim=0)
#                     t_feats_pt = torch.cat([t_feats_pt, t_feats.detach().cpu()], dim=0)
                    
                    criterion = nn.CrossEntropyLoss()
                    loss_cls = criterion(output, label)           
                    final_loss = loss_cls
                    losses.update(final_loss.item(), batch_size_cur)
                    acc1, acc5 = accuracy(output, label, topk=(1, 5))
                    
                    

                top1.update(acc1[0], batch_size_cur)
                top5.update(acc5[0], batch_size_cur)

#             计算类别准确率
#             ipdb.set_trace()
#             predicted = torch.max(output, 1)[1]
#             c = (predicted == label).squeeze()
#             for i in range(len(label)):
#                 label_i = label[i]
#                 class_correct[int(label_i)] += c[i].item()
#                 total_correct[int(label_i)] += 1
#         for i in range(172):
#             class_acc[i] = class_correct[i] * 100 / total_correct[i]
#         sortdict = sorted(class_acc.items(), key=lambda d: d[1], reverse=False)
#         for i in range(30):
#             test_log.write('Top{}/10 worst class:{}---{:.3f}\n'.format(i+1,sortdict[i][0], sortdict[i][1]))
#         for i in range(1,20,1):
#             test_log.write('Top{}/10 best class:{}---{:.3f}\n'.format(i+1,sortdict[-i][0], sortdict[-i][1]))
           
                
                        

        if opt.contrastive ==True:                
            log_out = ('Epoch: {epoch} Results: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg:.5f} Loss_con {loss_con.avg:.4f} Loss_cls1 {loss_cls1.avg:.4f} Loss_cls2 {loss_cls2.avg:.4f} Time {time:.3f}'
                  .format(epoch=epoch, top1=top1, top5=top5, loss=losses, loss_con=losses_con,loss_cls1=losses_cls1,loss_cls2=losses_cls2,time=round((time.time() - start_time), 4)))
            print(log_out)
            test_log.write(log_out + '\n')
            test_log.flush()
        else:
            log_out = ('Epoch: {epoch} Results: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg:.5f} Time {time:.3f}'
                  .format(epoch=epoch, top1=top1, top5=top5, loss=losses, time=round((time.time() - start_time), 4)))
            print(log_out)
            test_log.write(log_out + '\n')
            test_log.flush()
        return top1.avg
            

def lr_scheduler(epoch, optimizer, lr_decay_iter, decay_rate):
    if not (epoch % lr_decay_iter):
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = optimizer.param_groups[i]['lr'] * decay_rate
       
            
if __name__ == '__main__':
    log_training = open(os.path.join(result_path, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(result_path, 'log_test.csv'), 'w')

    for epoch in range(1, EPOCHS + 1):
        if opt.stage==3 and opt.method=='clip':
            lr_scheduler(epoch, optimizer_img, opt.lr_decay, opt.lrd_rate_finetune)
            lr_scheduler(epoch, optimizer_word, opt.lr_decay, opt.lrd_rate_finetune)
            lr_scheduler(epoch, optimizer_cls, opt.lr_decay, opt.lrd_rate)
            optimizer = [optimizer_img, optimizer_word, optimizer_cls]
        elif opt.method in ['recon','img2word']:
            lr_scheduler(epoch, optimizer_finetune, opt.lr_decay, opt.lrd_rate_finetune)
            lr_scheduler(epoch, optimizer_tune, opt.lr_decay, opt.lrd_rate)
            optimizer = [optimizer_tune,optimizer_finetune]
        else:
# frozen/no frozen
#             lr_scheduler(epoch, optimizer, opt.lr_decay, opt.lrd_rate)
# divided
            lr_scheduler(epoch, optimizer_m1, opt.lr_decay, opt.lrd_rate)
            lr_scheduler(epoch, optimizer_m2, opt.lr_decay, opt.lrd_rate)
            optimizer = [optimizer_m1,optimizer_m2]

        if opt.test_only == False:
            train_epoch(epoch, opt.lr_decay, optimizer, opt.stage, log_training)
        measure_cur = test_epoch(epoch, opt.stage, log_testing)
# save current model
        if opt.contrastive == True:
            if measure_cur > measure_best:
                torch.save(model.state_dict(), result_path + '/model_best.pt')
                measure_best = measure_cur
#             torch.save(model.state_dict(), result_path + '/model_{}.pt'.format(epoch))

        if epoch==EPOCHS:
            if opt.test_only == False:
                torch.save(model.state_dict(), result_path + '/model_final.pt')




