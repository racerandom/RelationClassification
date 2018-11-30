# coding=utf-8
import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
import torch.utils.data as Data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def is_best_score(score, best_score, monitor):
    if not best_score:
        is_best = True
        best_score = score
    else:
        is_best = bool(score < best_score) if monitor.endswith('_loss') else bool(score > best_score)
        best_score = score if is_best else best_score
    return is_best, best_score


def save_checkpoint(state, is_best, filename):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        torch.save(state, filename)  # save checkpoint
        return "=> Saving a new best"
    else:
        return ""

def batch_to_device(data, device):
    """
    copy input list into device for GPU computation
    :param data:
    :param device:
    :return:
    """
    device_inputs = []
    for d in data:
        device_inputs.append(d.to(device=device))
    return device_inputs

def copyData2device(data, device):
    feat_dict, target = data
    feat_types = list(feat_dict.keys())
    feat_list = batch_to_device(list(feat_dict.values()), device)
    target = target.to(device=device)
    return dict(zip(feat_types, feat_list)), target

class MultipleDatasets(Data.Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)

