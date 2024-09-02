# -*- coding: utf-8 -*-
"""
------ * Supplementary Material * ------ 
AAAI23 submission ID 9008:
    DR-Block: Convolutional Dense Reparameterization 
                for CNN Free Improvement
"""

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import apex
from apex import amp
from apex.parallel import DistributedDataParallel


parser = argparse.ArgumentParser(description='Supplementary Material for AAAI23 submission: DR-Block: Convolutional Dense Reparameterization for CNN Free Improvement')
parser.add_argument('--dataset',  metavar='DS', default='cifar100', choices=['cifar10', 'cifar100', 'imagenet'])
parser.add_argument('--data',
                    metavar='DIR',
                    default='/ImageNet2012',
                    help='Path to ImageNet dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='ResNet-18', choices=['ResNet-18'])
parser.add_argument('-t', '--blocktype', metavar='BLK', default='dr_block', choices=['base', 'dr_block'])
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=200,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-b',
                    '--batch-size',
                    default=128,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 6400), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p',
                    '--print-freq',
                    default=100,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--dist',
                    dest='dist',
                    action='store_true',
                    help='set distributed training')
parser.add_argument('--local_rank',
                    default=0,
                    type=int,
                    help='node rank for distributed training')


def reduce_mean(tensor, nprocs, args):
    rt = tensor.clone()
    if args.dist:
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255,
                                  0.406 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255,
                                 0.225 * 255]).cuda().view(1, 3, 1, 1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1-float(epoch)/float(num_epochs))**power
    for param_group in optimizer.param_groups:
        if epoch==0:
            tmp = param_group['lr']
        else:
            tmp = param_group['lr']
            tmp = tmp*(float(num_epochs-epoch)/float(num_epochs+1-epoch))**power
            param_group['lr'] = tmp
    return lr


def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    main_worker(args.local_rank, args.nprocs, args)


def main_worker(local_rank, nprocs, args):
    cudnn.benchmark = True
    best_acc1 = .0
    if args.dist:
        dist.init_process_group(backend='nccl')
    
    # dataset-specific settings
    if args.dataset == 'imagenet':
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        train_trans = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ])
        val_trans = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ])
        train_dataset = datasets.ImageFolder(
                    traindir,
                    train_trans)
        val_dataset = datasets.ImageFolder(
                    valdir,
                    val_trans)
        num_classes = 1000
    else:
        train_trans = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        val_trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        if args.dataset == 'cifar10':
            train_dataset = datasets.cifar.CIFAR10(root='cifar10', train=True, transform=train_trans, download=True)
            val_dataset = datasets.cifar.CIFAR10(root='cifar10', train=False, transform=val_trans, download=True)
            num_classes = 10
        elif args.dataset == 'cifar100':
            train_dataset = datasets.cifar.CIFAR100(root='cifar100', train=True, transform=train_trans, download=True)
            val_dataset = datasets.cifar.CIFAR100(root='cifar100', train=False, transform=val_trans, download=True)
            num_classes = 100
        else:
            raise NotImplementedError
    
    # define model
    if 'cifar' in args.dataset:
        from resnet_cifar import Network
    else:
        from resnet import Network
    # TODO You may modify the model settings for other architectures
    print("=> creating model {} for {}".format(args.arch, args.dataset))
    model = Network([2, 2, 2, 2], 'basic', args.blocktype, num_classes)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    torch.cuda.set_device(local_rank)
    model.cuda()
    if args.dist:
        model = apex.parallel.convert_syncbn_model(model).to(local_rank)

    # define criterion, optim and sched
    args.batch_size = int(args.batch_size / nprocs)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=1e-4)
    if args.dist:
        model, optimizer = amp.initialize(model, optimizer)
        model = DistributedDataParallel(model)
    if args.dataset == 'imagenet':
        IMAGENET_TRAINSET_SIZE = 1281167
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs * IMAGENET_TRAINSET_SIZE // args.batch_size // ngpus_per_node)
    else:  
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 120, 160], gamma=0.2)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            loc = 'cuda:{}'.format(args.local_rank)
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            ckpt = checkpoint['state_dict']
            model.load_state_dict(ckpt)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) @ Acc 1: {}"
                  .format(args.resume, checkpoint['epoch'], checkpoint['best_acc1']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading
    if args.dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, local_rank, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.dist:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, local_rank,
              args)
        lr_scheduler.step(epoch)
        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, local_rank, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if args.local_rank == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'acc1': acc1,
                    'acc5': acc5,
                    'best_acc1': best_acc1,
                }, is_best, filename='{}_{}_{}'.format(args.dataset, args.arch, args.blocktype))

    print("!!! Best Acc1: ", best_acc1)


def train(train_loader, model, criterion, optimizer, epoch, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    prefetcher = data_prefetcher(train_loader)
    images, target = prefetcher.next()
    i = 0
    while images is not None:
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if args.dist:
            torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs, args)
        reduced_acc1 = reduce_mean(acc1, args.nprocs, args)
        reduced_acc5 = reduce_mean(acc5, args.nprocs, args)

        losses.update(reduced_loss.item(), images.size(0))
        top1.update(reduced_acc1.item(), images.size(0))
        top5.update(reduced_acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.dist:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and local_rank==0:
            progress.display(i)

        i += 1

        images, target = prefetcher.next()


def validate(val_loader, model, criterion, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        prefetcher = data_prefetcher(val_loader)
        images, target = prefetcher.next()
        i = 0
        while images is not None:

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.dist:
                torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.nprocs, args)
            reduced_acc1 = reduce_mean(acc1, args.nprocs, args)
            reduced_acc5 = reduce_mean(acc5, args.nprocs, args)

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_acc1.item(), images.size(0))
            top5.update(reduced_acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and local_rank==0:
                progress.display(i)

            i += 1

            images, target = prefetcher.next()

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                    top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint'):
    torch.save(state, filename+'.pth.tar')
    if is_best:
        shutil.copyfile(filename+'.pth.tar', filename+'_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
