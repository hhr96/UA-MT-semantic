import os
from tqdm import tqdm
import argparse
import numpy as np
from numpy import append
from glob import glob
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils.metrics import iou_score, recall, specificity, calculate_metric_percase, adapted_rand_index, variation_of_info, betti_number
from networks import arch
from utils.util import str2bool,AverageMeter
from dataloaders.dataset import Dataset, TwoStreamBatchSampler

ARCH_NAMES = arch.__all__

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/ICA/test', help='Name of Experiment')
parser.add_argument('--model', type=str,  default='ICA_NestedUNet_woDS_UAMT_unlabel', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--img_ext', default='.png', help='image file extension')
parser.add_argument('--mask_ext', default='.png', help='mask file extension')
parser.add_argument('--dataset', type=str,  default='ICA', help='data_name')
parser.add_argument('--mask_channels', default=1, type=int, help='mask channels')


parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                    choices=ARCH_NAMES,
                    help='model architecture: ' +
                         ' | '.join(ARCH_NAMES) +
                         ' (default: NestedUNet)')
parser.add_argument('--num_workers', default=0, type=int)

parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
parser.add_argument('--input_channels', default=3, type=int, help='input channels')
parser.add_argument('--deep_supervision', default=False, type=str2bool)
parser.add_argument('--input_w', default=512, type=int, help='image width')
parser.add_argument('--input_h', default=512, type=int, help='image height')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
test_data_path = "../data/" + args.dataset + "/test/"
snapshot_path = "../models/"+args.model+"/fold1/"
test_save_path = "../models/"+args.model+"/prediction/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

#data loading part

#run
def test_calculate_metric(epoch_num):
    cudnn.benchmark=True

    print("=> creating model %s" % args.arch)
    model = arch.__dict__[args.arch](args.num_classes, args.input_channels, has_dropout=False)

    model= model.cuda()

    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    model.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    model.eval()

    test_img_ids = glob(os.path.join(test_data_path, 'images', '*' + args.img_ext))
    test_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_img_ids]

    test_transform = Compose([
        transforms.Resize(args.input_h, args.input_w),
        transforms.Normalize(),
    ])

    test_dataset = Dataset(
        img_list=test_img_ids,
        img_dir=os.path.join(test_data_path, 'images'),
        mask_dir=os.path.join(test_data_path, 'masks'),
        img_ext=args.img_ext,
        mask_ext=args.mask_ext,
        num_classes=args.mask_channels,
        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)
    iou = 0
    iou_a = []
    rec = 0
    rec_a = []
    over_a = []
    under_a = []
    betti_a = []
    spec = 0
    asd = 0
    hd = 0
    over = 0
    under = 0
    R_I = 0
    adapted_p = 0
    adapted_r = 0
    betti_num = 0

    for c in range(args.mask_channels):
        os.makedirs(os.path.join(test_save_path, str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, input_gray, meta, in tqdm(test_loader, total=len(test_loader)):
            input = input.cuda()
            dim = input.size(0)
            target = target.cuda()
            input_gray = input_gray.cuda()

            sample_image = torch.cat(
                (torch.unsqueeze(input_gray[0, :, :, :], 0), torch.unsqueeze(target[0, :, :, :], 0)), 0)

            # compute output
            if args.deep_supervision:
                outputs = model(input)
                output = output[-1]
                for layer in range(0, len(outputs)):
                    sample_image = torch.cat(
                        (sample_image, torch.unsqueeze(outputs[layer][0, :, :, :], 0)), 0)
                for i in range(1, dim):
                    sample_image = torch.cat((sample_image, torch.unsqueeze(input_gray[i, :, :, :], 0),
                                              torch.unsqueeze(target[i, :, :, :], 0)), 0)
                    for layer in range(0, len(outputs)):
                        sample_image = torch.cat(
                            (sample_image, torch.unsqueeze(outputs[layer][i, :, :, :], 0)), 0)
                sample_image = torchvision.utils.make_grid(sample_image, 6, 0)
                torchvision.utils.save_image(sample_image, os.path.join(test_save_path, str(c), meta['img_id'][i] + '_comb.png'))
            else:
                output = model(input)
                sample_image = torch.cat(
                    (sample_image, torch.unsqueeze(output[0, 0, :, :], 0).unsqueeze(1)), 0)
                for i in range(1, dim):
                    sample_image = torch.cat((sample_image, torch.unsqueeze(input_gray[i, :, :, :], 0),
                                              torch.unsqueeze(target[i, :, :, :], 0),
                                              torch.unsqueeze(output[i, 0, :, :], 0).unsqueeze(1)), 0)
                sample_image = torchvision.utils.make_grid(sample_image, 3, 0)
                torchvision.utils.save_image(sample_image, os.path.join(test_save_path, str(c), meta['img_id'][i] + '_comb.png'))

            iou = iou_score(torch.unsqueeze(output[:, 0], 1), target)+iou
            iou_a.append((2*iou_score(torch.unsqueeze(output[:, 0], 1), target))/(iou_score(torch.unsqueeze(output[:, 0], 1), target)+1))
            rec_temp, rec_a_t = recall(torch.unsqueeze(output[:, 0], 1), target)
            rec_a.append(rec_a_t)
            spec = specificity(torch.unsqueeze(output[:, 0], 1), target)+spec
            asd_temp, hd_temp = calculate_metric_percase(torch.unsqueeze(output[:, 0], 1), target)

            over_temp, under_temp, over_a_t, under_a_t = variation_of_info(torch.unsqueeze(output[:, 0], 1), target)
            over_a.append(over_a_t)
            under_a.append(under_a_t)
            R_I_temp, adapted_p_temp, adapted_r_temp = adapted_rand_index(torch.unsqueeze(output[:, 0], 1), target)
            betti_num_temp, betti_a_t = betti_number(torch.unsqueeze(output[:, 0], 1), target)
            betti_a.append(betti_a_t)

            rec= rec_temp+rec
            asd = asd_temp+asd
            hd = hd_temp+hd
            over = over_temp+over
            under = under_temp+under
            R_I = R_I_temp+R_I
            adapted_p = adapted_p_temp+adapted_p
            adapted_r = adapted_r_temp+adapted_r
            betti_num = betti_num_temp + betti_num

            # output = torch.sigmoid(output).cpu().numpy()
            #
            for i in range(dim):
                for c in range(args.mask_channels):
                    torchvision.utils.save_image(output[i, c], os.path.join(test_save_path, str(c), meta['img_id'][i] + '.png'))

    return iou/len(test_loader), np.std(iou_a), rec/len(test_loader), np.std(rec_a), \
           over / len(test_loader), np.std(over_a), under /len(test_loader), np.std(under_a), betti_num / len(test_loader), np.std(betti_a)


if __name__ == '__main__':
    metric1, sd1, metric2, sd2, metric3, sd3, metric4, sd4, metric5, sd5 = test_calculate_metric(1001)
    print('IoU: %.4f' % (metric1), 'sd: %.4f'%(sd1))
    print('rec: %.4f' % (metric2), 'sd: %.4f'%(sd2))
    print('over: %.4f' % (metric3), 'sd: %.4f'%(sd3))
    print('under: %.4f' % (metric4), 'sd: %.4f'%(sd4))
    print('betti: %.4f' % (metric5), 'sd: %.4f'%(sd5))