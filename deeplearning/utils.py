import os
import time
import copy

from opacus import PrivacyEngine

import torch
import torch.nn as nn
from ptflops import get_model_complexity_info

from deeplearning.nets.initialize import nn_registry

def init_all(config, dataset, N, logger):
    network = nn_registry[config.model](dataset.channel, dataset.num_classes, dataset.im_size, pretrained=config.pretrained)
    network = network.to(config.device)
    network.train()

    criterion = nn.CrossEntropyLoss().to(config.device)

    # Optimizer
    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(network.parameters(), config.lr, momentum=config.momentum,
                                    weight_decay=config.weight_decay, nesterov=config.nesterov)
    elif config.optimizer == "Adam":
        optimizer = torch.optim.Adam(network.parameters(), config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(network.parameters(), config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.__dict__[config.optimizer](network.parameters(), config.lr, momentum=config.momentum,
                                                            weight_decay=config.weight_decay, nesterov=config.nesterov)

    # LR scheduler
    if config.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N * config.epochs,
                                                                eta_min=config.min_lr)
    elif config.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=N * config.step_size,
                                                    gamma=config.gamma)
    else:
        scheduler = torch.optim.lr_scheduler.__dict__[config.scheduler](optimizer)

        # load checkpoint if the path exists
    if os.path.exists(config.checkpoint_path):
        checkpoint = torch.load(config.checkpoint_path)
        network.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
        start_epoch = checkpoint['epoch']
        # maxnorms = checkpoint['maxnorms']
        maxnorms = []
        logger.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(config.checkpoint_path, checkpoint['epoch']))
    else:
        start_epoch = 0

    # macs, params = get_model_complexity_info(network, (dataset.channel, config.im_size[0], config.im_size[1]), print_per_layer_stat=False, as_strings=True, verbose=False)
    # logger.info('{:<30}  {:<8}'.format('MACs: ', macs))
    # logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))

    return network, criterion, optimizer, scheduler, start_epoch

def init_dp_setup(config, network, dp_loader, N):
    if not config.use_sensitive:
        return None, None, None, [], None

    dp_model = copy.deepcopy(network)
    dp_model.train()

    # Optimizer
    if config.dp_optimizer == "SGD":
        optimizer = torch.optim.SGD(dp_model.parameters(), config.dp_lr, momentum=config.dp_momentum,
                                    weight_decay=config.dp_weight_decay, nesterov=config.dp_nesterov)
    elif config.optimizer == "Adam":
        optimizer = torch.optim.Adam(dp_model.parameters(), config.dp_lr, weight_decay=config.dp_weight_decay)
    else:
        optimizer = torch.optim.__dict__[config.optimizer](dp_model.parameters(), config.dp_lr, momentum=config.dp_momentum,
                                                            weight_decay=config.dp_weight_decay, nesterov=config.dp_nesterov)

    # LR scheduler
    if config.scheduler == "CosineAnnealingLR":
        dp_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N * config.epochs,
                                                                eta_min=config.min_lr)
    elif config.scheduler == "StepLR":
        dp_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=N * config.step_size,
                                                    gamma=config.gamma)
    else:
        dp_scheduler = torch.optim.lr_scheduler.__dict__[config.scheduler](optimizer)

    if config.dp_on:
        privacy_engine = PrivacyEngine()
        dp_model, dp_optimizer, dploader = privacy_engine.make_private(
            module = dp_model,
            optimizer = optimizer,
            data_loader = dp_loader,
            noise_multiplier = config.std,
            max_grad_norm = config.bound,
            poisson_sampling = False)
    else:
        privacy_engine = None
        dp_optimizer = optimizer

    return dp_model, dp_optimizer, dp_scheduler, dp_loader, privacy_engine


def init_optimizer( config, network, N):
    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(network.parameters(), 0.1*config.lr, momentum=config.momentum,
                                    weight_decay=config.weight_decay, nesterov=config.nesterov)
    elif config.optimizer == "Adam":
        optimizer = torch.optim.Adam(network.parameters(), config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.__dict__[config.optimizer](network.parameters(), config.lr, momentum=config.momentum,
                                                            weight_decay=config.weight_decay, nesterov=config.nesterov)

    if config.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N * config.epochs,
                                                                eta_min=config.min_lr)
    elif config.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=N * config.step_size,
                                                    gamma=config.gamma)
    else:
        scheduler = torch.optim.lr_scheduler.__dict__[config.scheduler](optimizer)

    return optimizer, scheduler

def save_checkpoint(state, path):
    torch.save(state, path)

def train(train_loader, network, criterion, optimizer, scheduler, epoch, config, logger, record):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracy_meter = AverageMeter('Accs', ':6.2f')

    # switch to train mode
    network.train()

    end = time.time()
    for i, contents in enumerate(train_loader):
        optimizer.zero_grad()
        target = contents[1].to(config.device)
        input = contents[0].to(config.device)

        # Compute output
        output = network(input)
        loss = criterion(output, target).mean()

        # Measure accuracy and record loss
        acc = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        accuracy_meter.update(acc.item(), input.size(0))

        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i == 0 or i == len(train_loader)-1:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, accuracy=accuracy_meter))
            
            record["train_loss"].append(losses.avg)
            record["train_acc"].append(accuracy_meter.avg)

    record["train_acc"].append(accuracy_meter.avg)


def test(test_loader, network, criterion, config, logger, record):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracy_meter = AverageMeter('Accs', ':6.2f')

    # Switch to evaluate mode
    network.eval()
    network.no_grad = True

    end = time.time()
    for i, data in enumerate(test_loader):
        input, target = data[0], data[1]

        target = target.to(config.device)
        input = input.to(config.device)

        # Compute output
        with torch.no_grad():
            output = network(input)

            loss = criterion(output, target).mean()

        # Measure accuracy and record loss
        acc = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        accuracy_meter.update(acc.item(), input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


    logger.info('Test acc: * Acc {accuracy.avg:.3f}'.format(accuracy=accuracy_meter))
    
    record["test_loss"].append(losses.avg)
    record["test_acc"].append(accuracy_meter.avg)

    network.no_grad = False

    return accuracy_meter.avg

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


def accuracy(output, target, topk=(1,)):
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


