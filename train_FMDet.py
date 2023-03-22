#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp

import torch
import yaml

import torchfcn_
from train_fcn32s import get_parameters
from train_fcn32s import git_hash
import models.model
import torch.optim as optim
here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument(
        '--max-iteration', type=int, default=1000000, help='max iteration'
    )
    parser.add_argument(
        '--lr', type=float, default=3.0e-4, help='learning rate',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0001, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.99, help='momentum',
    )

    args = parser.parse_args()

    # args.git_hash = git_hash()

    now = datetime.datetime.now()
    args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset

    root = osp.expanduser('/media/wanghao/wanghao/有丝分裂检测/数据集/MITOS12/TMI2022')
    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
    train_dataset=torchfcn_.datasets.mitos12.mitos(root, split='train')
    val_dataset=torchfcn_.datasets.mitos12.mitos(root, split='val')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4, shuffle=False, **kwargs)

    # 2. model

    model = models.model.FMDet()
    start_epoch = 0
    start_iteration = 0
    
    model = model.cuda()

    # 3. optimizer

    # optim = torch.optim.SGD(
    #     [
    #         {'params': get_parameters(model, bias=False)},
    #         {'params': get_parameters(model, bias=True),
    #          'lr': args.lr * 2, 'weight_decay': 0},
    #     ],
    #     lr=args.lr,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay)
    
    # optimizer=optim.SGD(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    optimizer=optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    # if args.resume:
    #     optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = torchfcn_.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        out=args.out,
        max_iter=args.max_iteration,
        interval_validate=2000,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
