"""
Title: Depth-induced Multi-scale Recurrent Attention Network for Saliency Detection
Author: Wei Ji, Jingjing Li
E-mail: weiji.dlut@gmail.com
"""
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision

import torch.nn.functional as F
import torch.optim as optim
from dataset_loader import MyData, MyTestData
from model import RGBNet,DepthNet
from fusion import ConvLSTM
from functions import imsave
import argparse
from trainer import Trainer

import os


configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    1: dict(
        max_iteration=1000000,
        lr=1.0e-10,
        momentum=0.99,
        weight_decay=0.0005,
        spshot=20000,
        nclass=2,
        sshow=10,
    )
}

parser=argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='test', help='train or test')
parser.add_argument('--param', type=str, default=True, help='path to pre-trained parameters')
# parser.add_argument('--train_dataroot', type=str, default='/home/jiwei-computer/Documents/Depth_data/train_data', help='path to train data')
parser.add_argument('--train_dataroot', type=str, default='/home/jiwei-computer/Documents/Depth_data/train_data-augment', help='path to train data')
parser.add_argument('--test_dataroot', type=str, default='/home/jiwei-computer/Documents/Depth_data/DUT-RGBD/test_data', help='path to test data')
# parser.add_argument('--test_dataroot', type=str, default='/home/jiwei-computer/Documents/Depth_data/NJUD/test_data', help='path to test data')
# parser.add_argument('--test_dataroot', type=str, default='/home/jiwei-computer/Documents/Depth_data/NLPR/test_data', help='path to test data')
# parser.add_argument('--test_dataroot', type=str, default='/home/jiwei-computer/Documents/Depth_data/LFSD', help='path to test data')
# parser.add_argument('--test_dataroot', type=str, default='/home/jiwei-computer/Documents/Depth_data/SSD', help='path to test data')
# parser.add_argument('--test_dataroot', type=str, default='/home/jiwei-computer/Documents/Depth_data/STEREO', help='path to test data')
# parser.add_argument('--test_dataroot', type=str, default='/home/jiwei-computer/Documents/Depth_data/RGBD135', help='path to test data')   # Need to set dataset_loader.py/line 113
parser.add_argument('--snapshot_root', type=str, default='./snapshot', help='path to snapshot')
parser.add_argument('--salmap_root', type=str, default='./sal_map', help='path to saliency map')
parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys())
args = parser.parse_args()
cfg = configurations[args.config]

cuda = torch.cuda.is_available()

"""""""""""~~~ dataset loader ~~~"""""""""

train_dataRoot = args.train_dataroot
test_dataRoot = args.test_dataroot

if not os.path.exists(args.snapshot_root):
    os.mkdir(args.snapshot_root)
if not os.path.exists(args.salmap_root):
    os.mkdir(args.salmap_root)

if args.phase == 'train':
    SnapRoot = args.snapshot_root           # checkpoint
    train_loader = torch.utils.data.DataLoader(MyData(train_dataRoot, transform=True),
                                               batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
else:
    MapRoot = args.salmap_root
    test_loader = torch.utils.data.DataLoader(MyTestData(test_dataRoot, transform=True),
                                   batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
print ('data already')
""""""""""" ~~~nets~~~ """""""""

start_epoch = 0
start_iteration = 0


model_rgb = RGBNet(cfg['nclass'])
model_depth = DepthNet(cfg['nclass'])
model_clstm = ConvLSTM(input_channels=64, hidden_channels=[64, 32, 64],
                 kernel_size=5, step=4, effective_step=[2, 4, 8])




if args.param is True:
    model_rgb.load_state_dict(torch.load(os.path.join(args.snapshot_root, 'snapshot_iter_1000000.pth')))
    model_depth.load_state_dict(torch.load(os.path.join(args.snapshot_root, 'depth_snapshot_iter_1000000.pth')))
    model_clstm.load_state_dict(torch.load(os.path.join(args.snapshot_root, 'clstm_snapshot_iter_1000000.pth')))
else:
    vgg19_bn = torchvision.models.vgg19_bn(pretrained=True)
    model_rgb.copy_params_from_vgg19_bn(vgg19_bn)
    model_depth.copy_params_from_vgg19_bn(vgg19_bn)


if cuda:
   model_rgb = model_rgb.cuda()
   model_depth = model_depth.cuda()
   model_clstm = model_clstm.cuda()


if args.phase == 'train':

    # Trainer: class, defined in trainer.py
    optimizer_rgb = optim.SGD(model_rgb.parameters(), lr=cfg['lr'],momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    optimizer_depth = optim.SGD(model_depth.parameters(), lr=cfg['lr'],momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    optimizer_clstm = optim.SGD(model_clstm.parameters(), lr=cfg['lr'],momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])

    training = Trainer(
        cuda=cuda,
        model_rgb=model_rgb,
        model_depth=model_depth,
        model_clstm=model_clstm,
        optimizer_rgb=optimizer_rgb,
        optimizer_depth=optimizer_depth,
        optimizer_clstm=optimizer_clstm,
        train_loader=train_loader,
        max_iter=cfg['max_iteration'],
        snapshot=cfg['spshot'],
        outpath=args.snapshot_root,
        sshow=cfg['sshow']
    )
    training.epoch = start_epoch
    training.iteration = start_iteration
    training.train()

else:

    for id, (data, depth, img_name, img_size) in enumerate(test_loader):
        print('testing bach %d' % (id+1))

        inputs = Variable(data).cuda()
        inputs_depth = Variable(depth).cuda()

        n, c, h, w = inputs.size()
        depth = inputs_depth.view(n, h, w, 1).repeat(1, 1, 1, c)
        depth = depth.transpose(3, 1)
        depth = depth.transpose(3, 2)


        h1, h2, h3, h4, h5 = model_rgb(inputs)  # RGBNet's output
        depth_vector, d1, d2, d3, d4, d5 = model_depth(depth)  # DepthNet's output
        outputs_all = model_clstm(depth_vector, h1, h2, h3, h4, h5, d1, d2, d3, d4, d5)  # Final output
        outputs_all = F.softmax(outputs_all, dim=1)
        outputs = outputs_all[0][1]
        outputs = outputs.cpu().data.resize_(h, w)

        imsave(os.path.join(MapRoot,img_name[0] + '.png'), outputs, img_size)

    print('The testing process has finished!')

