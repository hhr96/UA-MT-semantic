import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
from glob import glob
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from utils.metrics import iou_score, recall, specificity, calculate_metric_percase, adapted_rand_index, variation_of_info, betti_number
from networks import arch
from utils.util import str2bool,AverageMeter
from dataloaders import utils# image sampling 用的
from utils import ramps, losses
from dataloaders.dataset import Dataset, TwoStreamBatchSampler

ARCH_NAMES = arch.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES. append('BCEWithLogitsLoss')

#training setup
parser = argparse.ArgumentParser()
parser.add_argument('--name',  default=None, help='model_name')#模型的名字
parser.add_argument('--dataset', type=str,  default='ICA', help='data_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu') #要与下面的bs区分开
parser.add_argument('--labeled_bs', type=int, default=1, help='labeled_batch_size per gpu')
parser.add_argument('--img_ext', default='.png', help='image file extension')
parser.add_argument('--mask_ext', default='.png', help='mask file extension')

#model
parser.add_argument('--arch', '-a', metavar='ARCH', default='UNet',
                    choices=ARCH_NAMES,
                    help='model architecture: ' +
                         ' | '.join(ARCH_NAMES) +
                         ' (default: NestedUNet)')
parser.add_argument('--deep_supervision', default=False, type=str2bool)  # unet++有四层子网络，这个就是控制两种运行模式，一种是只看最外层的loss，一种是四层loss平均，但iou的计算都是只看最外层，所以对于模型的保存条件并没有不同
parser.add_argument('--input_channels', default=3, type=int, help='input channels')
parser.add_argument('--mask_channels', default=2, type=int, help='mask channels')
parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
parser.add_argument('--input_w', default=512, type=int, help='image width')
parser.add_argument('--input_h', default=512, type=int, help='image height')

parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')

parser.add_argument('--supervised_loss', default='BCEDiceLoss',
                    choices=LOSS_NAMES,
                    help='loss: ' +
                         ' | '.join(LOSS_NAMES) +
                         ' (default: BCEDiceLoss)')

#optimizer
parser.add_argument('--optimizer', default='SGD',
                    choices=['Adam', 'SGD'],
                    help='loss: ' +
                         ' | '.join(['Adam', 'SGD']) +
                         ' (default: Adam)')
parser.add_argument('--base_lr', type=float,  default=0.01, help='intial LR')
parser.add_argument('--momentum', default=0.9, type=float,  help='momentum')     # 沿着之前斜率先落一段，稍微快一点的sgd，但是如果很窄的minimum容易错过，因为先跳了之前的斜率
parser.add_argument('--nesterov', default=False, type=str2bool, help='nesterov') # 计算未来超前点的斜率，更快的sgd，过minimum会跳回来
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
                    # 当越接近最低值得时候，weight decay可以让更新的值变小， 更不容易出minimum https://towardsdatascience.com/why-adamw-matters-736223f31b5d, L2cost
parser.add_argument('--beta1', default=0.9, type=float, help='adam beta1')
parser.add_argument('--beta2', default=0.999, type=float, help='adam beta2')

# scheduler
parser.add_argument('--scheduler', default='CosineAnnealingLR',  # 四种lr的调整方式
                    choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
parser.add_argument('--min_lr', default=1e-8, type=float, help='minimum learning rate')  # lr按照cosine曲线趋势减少，减到min_lr为止
parser.add_argument('--factor', default=0.1, type=float)
parser.add_argument('--patience', default=2, type=int)  # Reduce learning rate when a metric has stopped improving. 这里就是两个epochs后模型无法提高，lr就乘上一个值降低
parser.add_argument('--milestones', default='1,2', type=str)  # lr按照milestones定的epoch时减小
parser.add_argument('--gamma', default=2 / 3, type=float)
parser.add_argument('--early_stopping', default=-1, type=int, metavar='N', help='early stopping (default: -1)')

parser.add_argument('--num_workers', default=0, type=int)
# added
parser.add_argument('--sample_freq', default=1, type=int, help='sample during validation')
parser.add_argument('--val_set_ratio', default=0.1, type=float, help='portion from training set to be validation set')
parser.add_argument('--energy_alpha', default=0.35, type=float, help='Energy loss function')
parser.add_argument('--energy_sigma', default=0.1, type=float, help='Energy loss function')

### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
args = parser.parse_args()

a = np.random.randint(-5,5,(5,5))

patch_size = (args.input_w, args.input_w, args.input_channels)
train_data_path = "../data/" + args.dataset + "/train/"
val_data_path = "../data/" + args.dataset + "/val/"
if args.name is None:
    if args.deep_supervision:
        args.name = '%s_%s_wDS_UAMT_unlabel' % (args.dataset, args.arch)
    else:
        args.name = '%s_%s_woDS_UAMT_unlabel' % (args.dataset, args.arch)
snapshot_path = "../models/" + args.name + "/" #pretrained model path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(iter_num, args, trainloader, model, ema_model, supervised_criterion, consistency_criterion, optimizer, lr_):
    model.train()

    for volume_batch, label_batch, _, _ in trainloader:
        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
        unlabeled_volume_batch = volume_batch[args.labeled_bs:]
        # compute output from network
        noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.0005, -0.0002, 0.0002)
        ema_inputs = unlabeled_volume_batch + noise
        outputs = model(volume_batch)
        with torch.no_grad():
            ema_output = ema_model(ema_inputs)
        # Monte Carlo Dropout
        T = 8
        volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1)  # 相当于两个unlabeled_volume_batch堆叠
        stride = volume_batch_r.shape[0] // 2  # //是整除
        preds = torch.zeros([stride * T, args.num_classes, args.input_w, args.input_h]).cuda()
        for i in range(T // 2):  # 做四次
            ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.0005, -0.0002, 0.0002)
            with torch.no_grad():
                preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs)
        preds = F.softmax(preds, dim=1)
        preds = preds.reshape(T, stride, args.num_classes, args.input_w, args.input_h)
        preds = torch.mean(preds, dim=0)  # (batch, 2, 112,112,80)？？为什么mean了就回到原来的size了
        uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1,
                                       keepdim=True)  # equation2 #(batch, 1, 112,112,80)

        ## calculate the loss
        # supervised_loss
        supervised_loss = 0
        if args.supervised_loss == 'EnergyLoss' or args.supervised_loss == 'EnergyLoss_new':
            output_temp = torch.sigmoid(outputs[:args.labeled_bs])
            for c in range(output_temp.shape[1]):
                score1 = output_temp[:, c, :, :]  # prob for class target 分布应该是0到1
                score2 = (0.5 - score1)
                supervised_loss = supervised_criterion(score2, torch.unsqueeze(
                    label_batch[:args.labeled_bs][:, c, :, :],1))+supervised_loss
        else:
            for c in range(outputs.shape[1]):
                supervised_loss = supervised_criterion(torch.unsqueeze(outputs[:args.labeled_bs, c], 1),
                                                   torch.unsqueeze(label_batch[:args.labeled_bs,c],1)) + supervised_loss
        # unsupervised_loss
        # control the balance between the supervised loss and unsupervised consistency loss
        consistency_weight = get_current_consistency_weight(iter_num // 150)
        consistency_dist = consistency_criterion(outputs[args.labeled_bs:],
                                                 ema_output)  # (batch, 2, 112,112,80) either KL divergence or voxel-level mean squared error
        threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, args.max_iterations)) * np.log(2)  # np.log(2) is the maximum uncertainty value
        mask = (uncertainty < threshold).float()
        consistency_dist = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)  # equation3
        consistency_loss = consistency_weight * consistency_dist
        # equation1
        loss = supervised_loss + consistency_loss
        iou = 0
        for c in range(outputs.shape[1]):
            iou = iou_score(torch.unsqueeze(outputs[:args.labeled_bs,c],1),
                            torch.unsqueeze(label_batch[:args.labeled_bs,c],1)) + iou

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, args.ema_decay, iter_num)

        iter_num = iter_num + 1
        writer.add_scalar('uncertainty/mean', uncertainty[0, 0].mean(), iter_num)
        writer.add_scalar('uncertainty/max', uncertainty[0, 0].max(), iter_num)
        writer.add_scalar('uncertainty/min', uncertainty[0, 0].min(), iter_num)
        writer.add_scalar('uncertainty/mask_per', torch.sum(mask) / mask.numel(), iter_num)
        writer.add_scalar('uncertainty/threshold', threshold, iter_num)
        writer.add_scalar('lr', lr_, iter_num)
        writer.add_scalar('loss/loss', supervised_loss, iter_num)
        writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
        writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
        writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)
        writer.add_scalar('train/iou', iou, iter_num)

        logging.info('iteration %d : loss : %f cons_dist: %f, loss_weight: %f' %
                     (iter_num, loss.item(), consistency_dist.item(), consistency_weight))

        ## change lr
        # if iter_num % 2500 == 0:
        #     lr_ = base_lr * 0.1 ** (iter_num // 2500)
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr_

        if iter_num >= args.max_iterations:
            break
    return iter_num

def validate(iter_num, args, valloader, model, ema_model, snapshot_path):
    model.eval()

    for volume_batch, label_batch, volume_gray, _ in valloader:
        volume_batch, label_batch, volume_gray = volume_batch.cuda(), label_batch.cuda(), volume_gray.cuda()

        outputs = model(volume_batch)
        iou, rec, betti_num = 0, 0, 0

        for c in range(outputs.shape[1]):
            iou = iou_score(outputs[:, c], label_batch[:, c]) + iou
            rec = recall(outputs[:, c], label_batch[:, c])+rec
            # spec = specificity(torch.unsqueeze(outputs[:, c], 1), torch.unsqueeze(label_batch[:, c], 1))+spec
            # asd, hd = calculate_metric_percase(torch.unsqueeze(outputs[:, 0], 1), label_batch)
            # over, under = variation_of_info(torch.unsqueeze(outputs[:, 0], 1), label_batch)
            # R_I, adapted_p, adapted_r= adapted_rand_index(torch.unsqueeze(outputs[:, 0], 1), label_batch)
            betti_num = betti_number(torch.unsqueeze(outputs[:, c], 1), torch.unsqueeze(label_batch[:, c], 1))+betti_num

        writer.add_scalar('val/iou', iou, iter_num)
        writer.add_scalar('val/recall', rec, iter_num)
        # writer.add_scalar('val/specificity', spec, iter_num)
        # writer.add_scalar('val/asd', asd, iter_num)
        # writer.add_scalar('val/hd95', hd, iter_num)
        # writer.add_scalar('val/over', over, iter_num)
        # writer.add_scalar('val/under', under, iter_num)
        # writer.add_scalar('val/R_I', R_I, iter_num)
        # writer.add_scalar('val/adapted_p', adapted_p, iter_num)
        # writer.add_scalar('val/adapted_r', adapted_r, iter_num)
        writer.add_scalar('val/betti_num', betti_num, iter_num)

        if iou > args.best_iou:  # 模型的保存条件 对比iou的最佳值
            save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            args.best_iou = iou
            print("=> saved best model")

        if np.mod(iter_num, args.sample_freq) == 0:
            sample_image = torch.cat((torch.unsqueeze(label_batch[0, 0, :, :], 0).unsqueeze(1)
                                      , torch.unsqueeze(outputs[0, 0, :, :], 0).unsqueeze(1)), 0)
            sample_image = torch.cat((sample_image, torch.unsqueeze(label_batch[0, 1, :, :], 0).unsqueeze(1)
                                      , torch.unsqueeze(outputs[0, 1, :, :], 0).unsqueeze(1)), 0)
            for i in range(1, label_batch.size(0)):
                for c in range (label_batch.size(1)):
                    sample_image = torch.cat((sample_image, torch.unsqueeze(label_batch[i, c, :, :], 0).unsqueeze(1),
                                            torch.unsqueeze(outputs[i, c, :, :], 0).unsqueeze(1)), 0)
            sample_image = torchvision.utils.make_grid(sample_image, label_batch.size(1)*2, 0)
            save_sample_path = os.path.join(snapshot_path+'sample/', 'iter_' + str(iter_num) + '.png')
            torchvision.utils.save_image(sample_image, save_sample_path)

        if iter_num >= args.max_iterations:
            break

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.makedirs(snapshot_path+"sample/")
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S') #logging the basic conficrations
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)) #日志输出到流，可以是sys.stderr，sys.stdout或者文件
    logging.info(str(args))

    #define loss function
    if args.consistency_type == 'mse':
        consistency_criterion = losses.__dict__['Softmaxmse']().cuda()
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.__dict__['Softmaxkl']().cuda()
    else:
        assert False, args.consistency_type
    if args.supervised_loss == 'BCEWithLogitsLoss':  # https://blog.csdn.net/yyhhlancelot/article/details/104260794
        supervised_criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        if args.supervised_loss == 'EnergyLoss' or args.supervised_loss == 'EnergyLoss_new':  # https://blog.csdn.net/yyhhlancelot/article/details/104260794
            supervised_criterion = losses.__dict__[args.supervised_loss](cuda=True, alpha=args.energy_alpha,sigma=args.energy_sigma)
        else:
            supervised_criterion = losses.__dict__[args.supervised_loss]().cuda()

    #define model type
    def create_model(ema=False):
        # Network definition
        print("=> creating model %s" % args.arch)
        net = arch.__dict__[args.arch](args.num_classes, args.input_channels, args.deep_supervision, has_dropout=True)  # 建立model的方式，通过引用arch.py里的class
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_() #当我们再训练网络的时候可能希望保持一部分的网络参数不变，只对其中一部分的参数进行调整；或者值训练部分分支网络，并不让其梯度对主网络的梯度造成影响，这时候我们就需要使用detach()函数来切断一些分支的反向传播
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    #要补上optimizer跟scheduler
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # params = filter(lambda p: p.requires_grad, model.parameters())    # transfer learning

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(), lr=base_lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=args.momentum,
                              nesterov=args.nesterov, weight_decay=args.weight_decay)  # 设定是什么optimizer 要么是adam要么是sgd
    else:
        raise NotImplementedError


    #dataloading
    train_img_ids = glob(os.path.join(train_data_path, 'images', '*' + args.img_ext))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]  # 将图像的名字从文件里取出
    val_img_ids = glob(os.path.join(val_data_path, 'images', '*' + args.img_ext))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]
    unlabeled_img_ids = glob(os.path.join(train_data_path, 'unlabeled', '*' + args.img_ext))
    unlabeled_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in unlabeled_img_ids]
    # train_img_ids, val_img_ids = train_test_split(img_ids, test_size=args.val_set_ratio,
    #                                               random_state=41)  # 选取validation image

    train_transform = Compose([
        transforms.RandomRotate90(),
        transforms.Flip(),
        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightness(),
            transforms.RandomContrast(),
        ], p=1),
        transforms.Resize(args.input_h, args.input_w),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        transforms.Resize(args.input_h, args.input_w),
        transforms.Normalize(),
    ])  # augmentation设定

    all_img_ids = list(set(train_img_ids) ^ set(unlabeled_img_ids)) + unlabeled_img_ids

    train_dataset = Dataset(
        img_list=all_img_ids,
        img_dir=os.path.join(train_data_path, 'images'),
        mask_dir=os.path.join(train_data_path, 'masks'),
        img_ext=args.img_ext,
        mask_ext=args.mask_ext,
        num_classes=args.mask_channels,
        transform=train_transform)
    val_dataset = Dataset(
        img_list=val_img_ids,
        img_dir=os.path.join(val_data_path, 'images'),
        mask_dir=os.path.join(val_data_path, 'masks'),
        img_ext=args.img_ext,
        mask_ext=args.mask_ext,
        num_classes=args.mask_channels,
        transform=val_transform)
    labeled_idxs = list(range((len(all_img_ids) - len(unlabeled_img_ids))))
    unlabeled_idxs = list(range(len(all_img_ids) - len(unlabeled_img_ids), len(all_img_ids)))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)

    # def worker_init_fn(worker_id):
    #     random.seed(args.seed+worker_id)

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    GLOBAL_WORKER_ID = None

    def worker_init_fn(worker_id):
        global GLOBAL_WORKER_ID
        GLOBAL_WORKER_ID = worker_id
        set_seed(args.seed + worker_id)
    trainloader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    max_epoch = max_iterations//len(trainloader)+1

    if args.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epoch, eta_min=args.min_lr)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience,
                                                   verbose=True, min_lr=args.min_lr)
    elif args.scheduler == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in args.milestones.split(',')],
                                             gamma=args.gamma)
    elif args.scheduler == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    model.train()
    ema_model.train()

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    #training begins
    iter_num = 0
    args.best_iou = 0
    lr_ = base_lr
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        iter_num = train(iter_num, args, trainloader, model, ema_model, supervised_criterion, consistency_criterion, optimizer, lr_)
        validate(iter_num, args, valloader, model, ema_model, snapshot_path)
        if args.scheduler == 'CosineAnnealingLR':
            scheduler.step()
            lr_ = optimizer.param_groups[0]['lr']
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(args.best_iou)
            lr_ = optimizer.param_groups[0]['lr']
        if iter_num >= max_iterations:
            break
    # save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    # torch.save(model.state_dict(), save_mode_path)
    # logging.info("save model to {}".format(save_mode_path))
    writer.close()
