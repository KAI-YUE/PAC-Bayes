import time
import copy
import numpy as np
from itertools import zip_longest

import torch
# from functorch import make_functional_with_buffers, vmap, grad

from deeplearning.utils import accuracy, AverageMeter
from deeplearning.datasets import fetch_dploader

def make_functional_with_buffers(mod, disable_autograd_tracking=False):
    params_dict = dict(mod.named_parameters())
    params_names = params_dict.keys()
    params_values = tuple(params_dict.values())

    buffers_dict = dict(mod.named_buffers())
    buffers_names = buffers_dict.keys()
    buffers_values = tuple(buffers_dict.values())
    
    stateless_mod = copy.deepcopy(mod)
    stateless_mod.to('meta')

    def fmodel(new_params_values, new_buffers_values, *args, **kwargs):
        new_params_dict = {name: value for name, value in zip(params_names, new_params_values)}
        new_buffers_dict = {name: value for name, value in zip(buffers_names, new_buffers_values)}
        return torch.func.functional_call(stateless_mod, (new_params_dict, new_buffers_dict), args, kwargs)
  
    if disable_autograd_tracking:
        params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)
    return fmodel, params_values, buffers_values


def clip_grad(model, max_norm, bound):
    if len(max_norm) == 0:
        print("======== init max_norm with a fixed value =========")
        for i, p in enumerate(model.parameters()):
            if p.grad is None:
                continue
            grad_norm = torch.norm(p.grad)
            if grad_norm > bound:
                p.grad *= bound / grad_norm
            max_norm.append(bound)
    else:
        j = 0
        for i, p in enumerate(model.parameters()):
            if p.grad is None:
                continue
            if max_norm[j] > bound:
                max_norm[j] = bound
            grad_norm = torch.norm(p.grad)
            if grad_norm > max_norm[j]:
                p.grad *= max_norm[j] / grad_norm
            j += 1

def add_noise(model, max_norm, batch_size, std, device):
    j = 0
    for i, p in enumerate(model.parameters()):
        if p.grad is None:
            continue
        noise = torch.normal(mean=0, std=std, size=p.shape).to(device)
        p.grad += max_norm[j] / batch_size * noise
        j += 1

def fetch_grad(model):
    grads = []
    for p in model.parameters():
        grads.append(p.grad)
    return grads

def mydp_train(train_loader, dptrain_loader, network, criterion, optimizer, scheduler, epoch, config, logger, record):
    """Train for one epoch on the training set with DP.
    The clipping bound will be dynamically computed on the first epoch (?)
    """

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracy_meter = AverageMeter('Accs', ':6.2f')

    # switch to train mode
    network.train()

    max_norms = []
    for i, (sens_data, insens_data) in enumerate(zip_longest(dptrain_loader, train_loader)):
        
        if config.use_sensitive and config.include_sensitive_this_epoch and sens_data is not None:                
            optimizer.zero_grad()
            
            if not config.dp_on:
                input = sens_data[0].to(config.device)
                label = sens_data[1].to(config.device)

                output = network(input)
                loss = criterion(output, label)
                
                loss.backward()

                optimizer.step()
                scheduler.step()
                losses.update(loss.data.item(), input.size(0))
                acc = accuracy(output.data, label, topk=(1,))[0]
                accuracy_meter.update(acc.item(), input.size(0))

                if i % config.print_every == 0:
                    logger.info('non-dp sens Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Acc {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        loss=losses, accuracy=accuracy_meter))
            else:
                for j, input in enumerate(sens_data[0]):
                    input = input.unsqueeze_(0).to(config.device)
                    label = sens_data[1][j].unsqueeze_(0).to(config.device)
                    
                    output = network(input)
                    loss = criterion(output, label)

                    loss.backward()

                    clip_grad(network, max_norms, bound=config.bound)
                    add_noise(network, max_norms, batch_size=config.batch_size, std=config.std, device=config.device)

                    losses.update(loss.data.item(), input.size(0))
                    acc = accuracy(output.data, label, topk=(1,))[0]
                    accuracy_meter.update(acc.item(), input.size(0))
                    # print("{:d} {:.3f}".format(j, losses.val))
                    
                    

                for _, p in enumerate(network.parameters()):
                    p.grad /= config.batch_size

                optimizer.step()

                if i % config.print_every == 0:
                    logger.info('sens Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Acc {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        loss=losses, accuracy=accuracy_meter))
                        
                    record["train_loss"].append(losses.avg)
                    record["train_acc"].append(accuracy_meter.avg)

                scheduler.step()

        if insens_data is not None:
            optimizer.zero_grad()
            input = insens_data[0].to(config.device)
            label = insens_data[1].to(config.device)

            output = network(input)
            loss = criterion(output, label)
            
            loss.backward()

            optimizer.step()
            scheduler.step()

            losses.update(loss.data.item(), input.size(0))
            acc = accuracy(output.data, label, topk=(1,))[0]
            accuracy_meter.update(acc.item(), input.size(0))

            if i % config.print_every == 0 or i == 77:
                logger.info('insens Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    loss=losses, accuracy=accuracy_meter))
                    
                record["train_loss"].append(losses.avg)
                record["train_acc"].append(accuracy_meter.avg)


def max_norm(subset, network, criterion, config):
    if not config.dp_on:
        return []

    # fmodel, params, buffers = make_functional_with_buffers(network)

    # def loss_fn(predictions, targets):
    #     return criterion(predictions, targets)
    
    # def compute_loss_stateless_model(params, buffers, sample, target):
    #     batch = sample.unsqueeze(0)
    #     targets = target.unsqueeze(0)

    #     predictions = fmodel(params, buffers, batch) 
    #     loss = loss_fn(predictions, targets)
    #     return loss
    
    # max_norms = []
    # in my own implementation, use one iteration to find the maximum norm,
    # then in the second iteration, add universal noise to the gradient that is proportional to the norm

    # config.batch_size = 20
    # small_batch_train_loader = fetch_dploader(config, subset)
    
    # for i, contents in enumerate(small_batch_train_loader):

    #     target = contents[1].to(config.device)
    #     input = contents[0].to(config.device)

        # ft_compute_grad = torch.func.grad(compute_loss_stateless_model)
        # ft_compute_sample_grad = torch.vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
        # ft_per_sample_grads = ft_compute_sample_grad(params, buffers, input, target)

        # # batch index
        # for j in range(len(ft_per_sample_grads[0])):
        #     # layer index
        #     for k in range(len(ft_per_sample_grads)):
        #         norm = torch.norm(ft_per_sample_grads[k][j])

        #         if i == 0 and j == 0:
        #             max_norms.append(norm.item())
        #         else:
        #             max_norms[k] = max(max_norms[k], norm.item())
        #         # print(norm)

        # break
        # for p in network.parameters():
        #     p.grad.zero_()

    data = torch.load("/mnt/ex-ssd/Projects/CV/indprivacy_train/model_zoo/0511_1418/checkpoint_epoch6.pth")
    maxnorms = data["maxnorms"]
    return maxnorms

    return max_norms


def mydp_jointtrain(train_loader, dptrain_loader, network, criterion, optimizer, scheduler, max_norms, epoch, config, logger, record):
    """Train for one epoch on the training set with DP.
    The clipping bound will be dynamically computed on the first epoch (?)
    """

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracy_meter = AverageMeter('Accs', ':6.2f')

    # switch to train mode
    network.train()

    end = time.time()

    # second iteration, do the actual training
    for i, (sens_data, insens_data) in enumerate(zip_longest(dptrain_loader, train_loader)):
        optimizer.zero_grad()
        if config.use_sensitive and config.include_sensitive_this_epoch and sens_data is not None:
            input = sens_data[0].to(config.device)
            label = sens_data[1].to(config.device)

            output = network(input)
            loss = criterion(output, label)
            
            loss.backward()

            if config.dp_on:
                clip_grad(network, max_norms, bound=config.bound)
                add_noise(network, max_norms, batch_size=config.batch_size, std=config.std, device=config.device)

            optimizer.step()
            scheduler.step()
            losses.update(loss.data.item(), input.size(0))
            acc = accuracy(output.data, label, topk=(1,))[0]
            accuracy_meter.update(acc.item(), input.size(0))

            if i % config.print_every == 0 or i == 77:
                logger.info('sens Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    loss=losses, accuracy=accuracy_meter))

        if insens_data is not None:
            optimizer.zero_grad()
            input = insens_data[0].to(config.device)
            label = insens_data[1].to(config.device)

            output = network(input)
            loss = criterion(output, label)
            
            loss.backward()

            optimizer.step()
            scheduler.step()
            losses.update(loss.data.item(), input.size(0))
            acc = accuracy(output.data, label, topk=(1,))[0]
            accuracy_meter.update(acc.item(), input.size(0))

        # if i == len(train_loader)-1 or i == len(dptrain_loader)-1: 
            if i % config.print_every == 0 or i == 77:
                logger.info('insens Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    loss=losses, accuracy=accuracy_meter))
                    
                record["train_loss"].append(losses.avg)
                record["train_acc"].append(accuracy_meter.avg)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # torch.cuda.empty_cache()


def interchange_train(
    sensitive_loader, insensitive_loader, 
    model, model_copy, criterion, 
    private_optimizer, optimizer, 
    private_scheduler, scheduler, privacy_engine,
    epoch, config, logger, record):
    """Train for one epoch on the sensitive and insensitive data
    """

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracy_meter = AverageMeter('Accs', ':6.2f')

    end = time.time()
    
    for i, (sens_data, insens_data) in enumerate(zip_longest(sensitive_loader, insensitive_loader)):
        if config.use_sensitive and config.include_sensitive_this_epoch and sens_data is not None:
            private_optimizer.zero_grad()
            # Training with sensitive data and private optimizer
            input = sens_data[0].to(config.device)
            label = sens_data[1].to(config.device)

            output = model_copy(input)
            loss = criterion(output, label)
            
            loss.backward()
            private_optimizer.step()
            private_scheduler.step()

            # Sync the parameters from the model_copy to the original model
            sync_parameters(model, model_copy)
            losses.update(loss.data.item(), input.size(0))

            # Measure accuracy and record loss
            acc = accuracy(output.data, label, topk=(1,))[0]
            accuracy_meter.update(acc.item(), input.size(0))
            
            if config.dp_on:
                epsilon = privacy_engine.get_epsilon(config.delta)
            else:
                epsilon = np.inf

            if i % config.print_every == 0 or i == 77:
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc {accuracy.val:.3f} ({accuracy.avg:.3f}) [sens]'.format(
                    epoch, i, len(sensitive_loader), batch_time=batch_time,
                    loss=losses, accuracy=accuracy_meter))
                logger.info("Epsilon: {:.2f}".format(epsilon))

        if config.use_insensitive and insens_data is not None:
            optimizer.zero_grad()
            # Training with public data and traditional optimizer
            input = insens_data[0].to(config.device)
            label = insens_data[1].to(config.device)

            output = model(input)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Sync the parameters from the original model to the model_copy
            sync_parameters(model_copy, model)
            losses.update(loss.data.item(), input.size(0))
        
            # Measure accuracy and record loss
            acc = accuracy(output.data, label, topk=(1,))[0]
            accuracy_meter.update(acc.item(), input.size(0))

            if i % config.print_every == 0 or i == 77:
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc {accuracy.val:.3f} ({accuracy.avg:.3f}) [insens]'.format(
                    epoch, i, len(insensitive_loader), batch_time=batch_time,
                    loss=losses, accuracy=accuracy_meter))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch: [{0}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Acc {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
        epoch, batch_time=batch_time,
        loss=losses, accuracy=accuracy_meter))
    logger.info("-"*80)

    record["train_loss"].append(losses.avg)


def sync_parameters(model1, model2):
    if model1 is None or model2 is None:
        return None
    with torch.no_grad():
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p1.copy_(p2)