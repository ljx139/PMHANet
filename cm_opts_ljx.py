import argparse

def opt_algorithm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default= 'wide',help='indicator to dataset')
    # path setting
    parser.add_argument('--result_path', type=str, default= '/mnt/results/',help='path to the folder to save results')
    
    # experiment controls
    # img -> 1
    # ingredient -> 2
    # final network -> 3
    parser.add_argument('--stage', type=int, default = 1,help='1: pretrain image channel independently; 2: pretrain ingredient channel independently; 3: train the whole network')
    parser.add_argument('--mode', type=str, default= 'train',help='select from train, val, test. Used in dataset creation')
    parser.add_argument('--img_net', type=str, default= '',help='choose network backbone for image channel: vgg19bn, resnet18, resnet50, wrn, wiser')
    parser.add_argument('--feats', type=str, default= 'cls_token',help='choose features from: cls_token, pool')
    parser.add_argument('--word_net', type=str, default= 'gru',help='choose network backbone for ingredient channel: gru, nn')
    parser.add_argument('--method', type=str, default= None ,help='choose method backbone in: align, clip, img2word')    
    
    # loss weight
    parser.add_argument('--pos_weight', type=float, default = 40, help='positive weight of bcelosswithlogits in nus-wide dataset')
    
    # img_lend
    # parser.add_argument('--img_lend', type=int, default= 512, help='feature dimension')
    
    # turning parameters
    
    parser.add_argument('--batch_size', type=int, default =64, help='batch size')
    parser.add_argument('--lr', type=float, default = 1e-2, help='learning rate')
    parser.add_argument('--lr_m1', type=float, default = 5e-5, help='learning rate for module 1')
    parser.add_argument('--lr_m2', type=float, default = 0, help='learning rate for module 2')
    parser.add_argument('--lr_finetune', type=float, default = 5e-5, help='fine-tune learning rate')
    parser.add_argument('--lrd_rate', type=float, default = 0.1, help='decay rate of learning rate')
    parser.add_argument('--lrd_rate_finetune', type=float, default = 0.1, help='decay rate of fine-tune learning rate')
    parser.add_argument('--lr_decay', type=int, default = 4, help='decay rate of learning rate')
    parser.add_argument('--weight_decay', type=float, default = 1e-3, help='weight decay')
    
    parser.add_argument('--beta_clip_align', type=float, default = 0.01, help='coefficient of kl loss between image features and word features')
    parser.add_argument('--beta_align', type=float, default = 0.1, help='coefficient of  l2norm loss between image features and word features')
    parser.add_argument('--type_align', type=str, default = 'l2', help='type of align loss: kl or l2')
    parser.add_argument('--beta_loss_word', type=float, default = 1.0, help='word prediction loss')
    parser.add_argument('--topk', type=int, default = 5,help='top num for img2word prediction')
    parser.add_argument('--beta_soft', type=float, default = 0,help='soft factor for class adj, from 0.0~1.0, the larger, the softer')
    parser.add_argument('--adj', type=str, default= 'gcn_class',help='recon_soft_sum')
    parser.add_argument('--gcn_time', type=int, default= 2,help='recon_soft_sum')
    parser.add_argument('--gcn_w2', type=float, default=1.,help='recon_soft_sum')
    parser.add_argument('--gcn_w3', type=float, default=1.,help='recon_soft_sum')
    parser.add_argument('--gcn_relation', type=float, default=0.5,help='relation factor of gcn,')
    parser.add_argument('--gcn_threshold', type=float, default=0.4,help='the bigger, the more different from exist tage realations, range from [0,0.5]')
    
    parser.add_argument('--pretrain_model', type=int, default =9,help='top num for img2word prediction')

#     ljx
    parser.add_argument('--frozen_blks', type=int, default =12,help='select i in 12 to freeze (0-i)blks')
    parser.add_argument('--exp', type=int, default =0,help='experiment idx')
    parser.add_argument('--test_only', type=bool, default = False ,help='test only')
    parser.add_argument('--contrastive', type=bool, default = False ,help='use contrastive learning')
    parser.add_argument('--VSaug', type=bool, default = False ,help='use V and S randomaugment')
    args = parser.parse_args()
    
    return args