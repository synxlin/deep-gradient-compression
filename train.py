import argparse
import math
import os
import random
import shutil

import numpy as np
import horovod.torch as hvd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from tqdm import tqdm

from torchpack.mtpack.utils.config import Config, configs

from dgc.horovod.optimizer import DistributedOptimizer
from dgc.compression import DGCCompressor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+')
    parser.add_argument('--devices', default='gpu')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--suffix', default='')
    args, opts = parser.parse_known_args()

    ##################
    # Update configs #
    ##################

    printr(f'==> loading configs from {args.configs}')
    Config.update_from_modules(*args.configs)
    Config.update_from_arguments(*opts)

    if args.devices is not None and args.devices != 'cpu':
        configs.device = 'cuda'
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        cudnn.benchmark = True
    else:
        configs.device = 'cpu'

    if 'seed' in configs and configs.seed is not None:
        random.seed(configs.seed)
        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        if configs.device == 'cuda' and configs.get('deterministic', True):
            cudnn.deterministic = True
            cudnn.benchmark = False
    
    configs.train.num_batches_per_step = \
        configs.train.get('num_batches_per_step', 1)

    configs.train.save_path = get_save_path(*args.configs) \
                              + f'{args.suffix}.np{hvd.size()}'
    printr(f'[train.save_path] = {configs.train.save_path}')
    checkpoint_path = os.path.join(configs.train.save_path, 'checkpoints')
    configs.train.checkpoint_path = os.path.join(
        checkpoint_path, f'e{"{epoch}"}-r{hvd.rank()}.pth'
    )
    configs.train.latest_pth_path = os.path.join(
        checkpoint_path, f'latest-r{hvd.rank()}.pth'
    )
    configs.train.best_pth_path = os.path.join(
        checkpoint_path, f'best-r{hvd.rank()}.pth'
    )
    os.makedirs(checkpoint_path, exist_ok=True)

    if args.evaluate:
        configs.train.latest_pth_path = configs.train.best_pth_path

    printr(configs)

    #####################################################################
    # Initialize DataLoaders, Model, Criterion, LRScheduler & Optimizer #
    #####################################################################

    printr(f'\n==> creating dataset "{configs.dataset}"')
    dataset = configs.dataset()
    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(configs.data.num_threads_per_worker)
    loader_kwargs = {'num_workers': configs.data.num_threads_per_worker,
                     'pin_memory': True} if configs.device == 'cuda' else {}
    # When supported, use 'forkserver' to spawn dataloader workers
    # instead of 'fork' to prevent issues with Infiniband implementations
    # that are not fork-safe
    if (loader_kwargs.get('num_workers', 0) > 0 and
            hasattr(mp, '_supports_context') and
            mp._supports_context and
            'forkserver' in mp.get_all_start_methods()):
        loader_kwargs['multiprocessing_context'] = 'forkserver'
    printr(f'\n==> loading dataset "{loader_kwargs}""')
    samplers, loaders = {}, {}
    for split in dataset:
        # Horovod: use DistributedSampler to partition data among workers.
        # Manually specify `num_replicas=hvd.size()` and `rank=hvd.rank()`.
        samplers[split] = torch.utils.data.distributed.DistributedSampler(
            dataset[split], num_replicas=hvd.size(), rank=hvd.rank())
        loaders[split] = torch.utils.data.DataLoader(
            dataset[split], batch_size=configs.train.batch_size * (
                configs.train.num_batches_per_step if split == 'train' else 1),
            sampler=samplers[split],
            drop_last=(configs.train.num_batches_per_step > 1
                       and split == 'train'),
            **loader_kwargs
        )

    printr(f'\n==> creating model "{configs.model}"')
    model = configs.model()
    model = model.cuda()

    criterion = configs.train.criterion().to(configs.device)
    # Horovod: scale learning rate by the number of GPUs.
    configs.train.base_lr = configs.train.optimizer.lr
    configs.train.optimizer.lr *= (configs.train.num_batches_per_step
                                   * hvd.size())
    printr(f'\n==> creating optimizer "{configs.train.optimizer}"')

    if configs.train.optimize_bn_separately:
        optimizer = configs.train.optimizer([
            dict(params=get_common_parameters(model)),
            dict(params=get_bn_parameters(model), weight_decay=0)
        ])
    else:
        optimizer = configs.train.optimizer(model.parameters())

    # Horovod: (optional) compression algorithm.
    printr(f'\n==> creating compression "{configs.train.compression}"')
    if configs.train.dgc:
        printr(f'\n==> initializing dgc compression')
        configs.train.compression.memory = configs.train.compression.memory()
        compression = configs.train.compression()
        compression.memory.initialize(model.named_parameters())
        cpr_parameters = {}
        for name, param in model.named_parameters():
            if param.dim() > 1:
                cpr_parameters[name] = param
        compression.initialize(cpr_parameters.items())
    else:
        compression = configs.train.compression()
    
    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression,
        backward_passes_per_step=configs.train.num_batches_per_step,
        op=hvd.Average
    )

    # resume from checkpoint
    last_epoch, best_metric = -1, None
    if os.path.exists(configs.train.latest_pth_path):
        printr(f'\n[resume_path] = {configs.train.latest_pth_path}')
        checkpoint = torch.load(configs.train.latest_pth_path)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint.pop('model'))
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint.pop('optimizer'))
        if configs.train.dgc and 'compression' in checkpoint:
            compression.memory.load_state_dict(checkpoint.pop('compression'))
        last_epoch = checkpoint.get('epoch', last_epoch)
        best_metric = checkpoint.get('meters', {}).get(
            f'{configs.train.metric}_best', best_metric)
        # Horovod: broadcast parameters.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    else:
        printr('\n==> train from scratch')
        # Horovod: broadcast parameters & optimizer state.
        printr('\n==> broadcasting paramters and optimizer state')
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    num_steps_per_epoch = len(loaders['train'])
    if 'scheduler' in configs.train and configs.train.scheduler is not None:
        if configs.train.schedule_lr_per_epoch:
            last = max(last_epoch - configs.train.warmup_lr_epochs - 1, -1)
        else:
            last = max((last_epoch - configs.train.warmup_lr_epochs + 1)
                       * num_steps_per_epoch - 2, -1)
        scheduler = configs.train.scheduler(optimizer, last_epoch=last)
    else:
        scheduler = None

    ############
    # Training #
    ############

    meters = evaluate(model, device=configs.device, meters=configs.train.meters,
                      loader=loaders['test'], split='test')
    for k, meter in meters.items():
        printr(f'[{k}] = {meter:2f}')
    if args.evaluate or last_epoch >= configs.train.num_epochs:
        return

    if hvd.rank() == 0:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(configs.train.save_path)
    else:
        writer = None

    for current_epoch in range(last_epoch + 1, configs.train.num_epochs):
        printr(f'\n==> training epoch {current_epoch}'
                f'/{configs.train.num_epochs}')

        if configs.train.dgc:
            compression.warmup_compress_ratio(current_epoch)

        train(model=model, loader=loaders['train'],
              device=configs.device, epoch=current_epoch, 
              sampler=samplers['train'], criterion=criterion,
              optimizer=optimizer, scheduler=scheduler,
              batch_size=configs.train.batch_size,
              num_batches_per_step=configs.train.num_batches_per_step,
              num_steps_per_epoch=num_steps_per_epoch,
              warmup_lr_epochs=configs.train.warmup_lr_epochs,
              schedule_lr_per_epoch=configs.train.schedule_lr_per_epoch,
              writer=writer, quiet=hvd.rank() != 0)

        meters = dict()
        for split, loader in loaders.items():
            if split != 'train':
                meters.update(evaluate(model, loader=loader,
                                       device=configs.device,
                                       meters=configs.train.meters,
                                       split=split, quiet=hvd.rank() != 0))

        best = False
        if 'metric' in configs.train and configs.train.metric is not None:
            if best_metric is None or best_metric < meters[configs.train.metric]:
                best_metric, best = meters[configs.train.metric], True
            meters[configs.train.metric + '_best'] = best_metric

        if writer is not None:
            num_inputs = ((current_epoch + 1) * num_steps_per_epoch
                          * configs.train.num_batches_per_step
                          * configs.train.batch_size * hvd.size())
            print('')
            for k, meter in meters.items():
                print(f'[{k}] = {meter:2f}')
                writer.add_scalar(k, meter, num_inputs)

        checkpoint = {
            'epoch': current_epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'meters': meters,
            'compression': compression.memory.state_dict() \
                            if configs.train.dgc else None
        }

        # save checkpoint
        checkpoint_path = \
            configs.train.checkpoint_path.format(epoch=current_epoch)
        torch.save(checkpoint, checkpoint_path)
        shutil.copyfile(checkpoint_path, configs.train.latest_pth_path)
        if best:
            shutil.copyfile(checkpoint_path, configs.train.best_pth_path)
        if current_epoch >= 3:
            os.remove(
                configs.train.checkpoint_path.format(epoch=current_epoch - 3)
            )
        printr(f'[save_path] = {checkpoint_path}')


def train(model, loader, device, epoch, sampler, criterion, optimizer,
          scheduler, batch_size, num_batches_per_step, num_steps_per_epoch, warmup_lr_epochs, schedule_lr_per_epoch, writer=None, quiet=True):
    step_size = num_batches_per_step * batch_size
    num_inputs = epoch * num_steps_per_epoch * step_size * hvd.size()
    _r_num_batches_per_step = 1.0 / num_batches_per_step

    sampler.set_epoch(epoch)
    model.train()
    for step, (inputs, targets) in enumerate(tqdm(
            loader, desc='train', ncols=0, disable=quiet)):
        adjust_learning_rate(scheduler, epoch=epoch, step=step,
                             num_steps_per_epoch=num_steps_per_epoch,
                             warmup_lr_epochs=warmup_lr_epochs,
                             schedule_lr_per_epoch=schedule_lr_per_epoch)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        loss = torch.tensor([0.0])
        for b in range(0, step_size, batch_size):
            _inputs = inputs[b:b+batch_size]
            _targets = targets[b:b+batch_size]
            _outputs = model(_inputs)
            _loss = criterion(_outputs, _targets)
            _loss.mul_(_r_num_batches_per_step)
            _loss.backward()
            loss += _loss.item()
        optimizer.step()

        # write train loss log
        loss = hvd.allreduce(loss, name='loss').item()
        if writer is not None:
            num_inputs += step_size * hvd.size()
            writer.add_scalar('loss/train', loss, num_inputs)


def evaluate(model, loader, device, meters, split='test', quiet=True):
    _meters = {}
    for k, meter in meters.items():
        _meters[k.format(split)] = meter()
    meters = _meters

    model.eval()

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=split, ncols=0, disable=quiet):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            for meter in meters.values():
                meter.update(outputs, targets)

    for k, meter in meters.items():
        data = meter.data()
        for dk, d in data.items():
            data[dk] = \
                hvd.allreduce(torch.tensor([d]), name=dk, op=hvd.Sum).item()
        meter.set(data)
        meters[k] = meter.compute()
    return meters


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning
# leads to worse final accuracy.
# Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()`
# during the first five epochs. See https://arxiv.org/abs/1706.02677.
def adjust_learning_rate(scheduler, epoch, step, num_steps_per_epoch,
                         warmup_lr_epochs=0, schedule_lr_per_epoch=False):
    if epoch < warmup_lr_epochs:
        size = hvd.size()
        epoch += step / num_steps_per_epoch
        factor = (epoch * (size - 1) / warmup_lr_epochs + 1) / size
        for param_group, base_lr in zip(scheduler.optimizer.param_groups,
                                        scheduler.base_lrs):
            param_group['lr'] = base_lr * factor
    elif schedule_lr_per_epoch and (step > 0 or epoch == 0):
        return
    elif epoch == warmup_lr_epochs and step == 0:
        for param_group, base_lr in zip(scheduler.optimizer.param_groups,
                                        scheduler.base_lrs):
            param_group['lr'] = base_lr
        return
    else:
        scheduler.step()

def get_bn_parameters(module):
    def get_members_fn(m):
        if isinstance(m, nn.BatchNorm2d):
            return m._parameters.items()
        else:
            return dict()
    gen = module._named_members(get_members_fn=get_members_fn)
    for _, elem in gen:
        yield elem


def get_common_parameters(module):
    def get_members_fn(m):
        if isinstance(m, nn.BatchNorm2d):
            return dict()
        else:
            for n, p in m._parameters.items():
                yield n, p

    gen = module._named_members(get_members_fn=get_members_fn)
    for _, elem in gen:
        yield elem


def get_save_path(*configs, prefix='runs'):
    memo = dict()
    for c in configs:
        cmemo = memo
        c = c.replace('configs/', '').replace('.py', '').split('/')
        for m in c:
            if m not in cmemo:
                cmemo[m] = dict()
            cmemo = cmemo[m]

    def get_str(m, p):
        n = len(m)
        if n > 1:
            p += '['
        for i, (k, v) in enumerate(m.items()):
            p += k
            if len(v) > 0:
                p += '.'
            p = get_str(v, p)
            if n > 1 and i < n - 1:
                p += '+'
        if n > 1:
            p += ']'
        return p

    return os.path.join(prefix, get_str(memo, ''))


def printr(*args, **kwargs):
    if hvd.rank() == 0:
        print(*args, **kwargs)


if __name__ == '__main__':
    hvd.init()
    main()
