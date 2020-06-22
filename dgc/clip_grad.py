import torch
from torch._six import inf

from horovod.torch import allreduce_

__all__ = ['clip_grad_norm_', 'clip_grad_value_', 'clip_grad_value_by_global_norm_', 'clip_grad_norm_2_by_global_']


# code modified from https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/clip_grad.py
def clip_grad_norm_(grad, max_norm, norm_type=2):
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = grad.data.abs().max()
    else:
        total_norm = grad.data.norm(norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        grad.data.mul_(clip_coef)
    return grad


def clip_grad_value_(grad, clip_value):
    clip_value = float(clip_value)
    grad.data.clamp_(min=-clip_value, max=clip_value)


# code modified from https://github.com/sands-lab/grace/blob/master/grace_dl/torch/memory/dgc.py
def clip_grad_value_by_global_norm_(grad, name=None):
    grad_square_sum = torch.sum(grad.square())
    clip_value = torch.sqrt(allreduce_(grad_square_sum, average=True, name=name))
    grad.data.clamp_(min=-clip_value, max=clip_value)


def clip_grad_norm_2_by_global_(grad, max_norm, name=None):
    max_norm = float(max_norm)
    grad_square_sum = torch.sum(grad.square())
    total_norm = torch.sqrt(allreduce_(grad_square_sum, average=True, name=name))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        grad.data.mul_(clip_coef)
    return grad
