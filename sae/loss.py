import torch
import numpy as np
from sae.util import scatter
from torch.nn.functional import cross_entropy, mse_loss
from scipy.optimize import linear_sum_assignment


def cross_entropy_loss(x1, x2):
    if len(x2.shape) == 2 and x2.shape[1] == 1:
        x2 = x2.squeeze(1)
    return cross_entropy(x1, x2, reduction='none')


def mean_squared_loss(x1, x2):
    return torch.mean(mse_loss(x1, x2, reduction='none'), dim=-1)


def get_loss_idxs(set1_lens, set2_lens):
    """Given two data of different sizes, return the indices for the
    first and second datas denoting the elements to be used in the loss
    function. setn_lens represents the number of elements in each batch.

    E.g. 
    >>> pred_idx, target_idx = get_loss_idxs(pred, target, ....)
    >>> loss = functional.mse_loss(pred[pred_idx], target[target_idx])
    """
    assert set1_lens.shape == set2_lens.shape
    set_lens = torch.min(set1_lens, set2_lens)

    ptr1_start = set1_lens.cumsum(dim=0).roll(1)
    ptr1_start[0] = 0
    ptr1_end = ptr1_start + set_lens

    ptr2_start = set2_lens.cumsum(dim=0).roll(1)
    ptr2_start[0] = 0
    ptr2_end = ptr2_start + set_lens

    ptr1 = torch.cat(
        [
            torch.arange(ptr1_start[i], ptr1_end[i], device=set1_lens.device) 
            for i in range(ptr1_start.numel())
        ]
    )
    ptr2 = torch.cat(
        [
            torch.arange(ptr2_start[i], ptr2_end[i], device=set2_lens.device) 
            for i in range(ptr2_start.numel())
        ]
    )
    return ptr1, ptr2


def correlation(x1, x2, return_arr=False):
    # finds the correlation coefficient between x1 and x2 across the batch, for each dimension
    # the output is the mean of the corrs for each dimension
    corr_mat = np.corrcoef(x1.cpu().detach(), x2.cpu().detach(), rowvar=False)
    n = x1.shape[-1]
    idxs = np.array([(i,i+n) for i in range(n)])
    try:
        corrs = corr_mat[idxs[:,0], idxs[:,1]]
        if return_arr:
            return torch.as_tensor(corrs)
        corr = np.mean(corrs)
        return corr
    except:
        return torch.nan


def batch_to_set_lens(batch, batch_size=None):
    return scatter(src=torch.ones(batch.shape, device=batch.device), index=batch, dim_size=batch_size).long()


def min_permutation_idxs(yhat, y, batch, loss_fn=cross_entropy_loss):
    n = batch_to_set_lens(batch)
    n = n[n != 0]
    ptr = torch.cat([torch.tensor([0], device=n.device), n], dim=0).cumsum(dim=0)
    perm = torch.empty(batch.shape, dtype=torch.long, device=ptr.device)
    for idx_start, idx_end in zip(ptr[:-1], ptr[1:]):
        yhati = yhat[idx_start:idx_end]
        yi = y[idx_start:idx_end]
        size = yi.shape[0]
        yhati_rep = yhati.repeat((size, 1))
        yi_rep = yi.repeat_interleave(size, dim=0)
        loss_pairwise = loss_fn(yhati_rep, yi_rep).view(size, size) # loss_pairwise[i,j] = loss_fn(yhati[j], yi[i])
        assignment = linear_sum_assignment(loss_pairwise.detach().cpu().numpy()) # lin_sum_ass(Cost) calculates the min assignment, given as (row_idxs, col_idxs)
        perm[idx_start:idx_end] = torch.tensor(assignment[1], device=perm.device) + idx_start
    return perm


def min_permutation_loss(yhat, y, batch, loss_fn=cross_entropy_loss):
    perm = min_permutation_idxs(yhat=yhat, y=y, batch=batch, loss_fn=loss_fn)
    loss = torch.mean(loss_fn(yhat[perm], y))
    return loss


def fixed_order_idxs(y, batch, order_fn=lambda y: y.float() @ torch.arange(1,y.shape[1]+1).float()):
    # order_fn: takes y (n x d) as input, outputs a scalar (n) to be used for sorting
    if order_fn is None:
        return slice(None) #torch.arange(batch.shape[0])
    n = batch_to_set_lens(batch)
    n = n[n != 0]
    ptr = torch.cat([torch.tensor([0]), n], dim=0).cumsum(dim=0)
    perm = torch.empty(batch.shape, dtype=torch.long)
    for idx_start, idx_end in zip(ptr[:-1], ptr[1:]):
        yi = y[idx_start:idx_end]
        rank = order_fn(yi)
        idxs = torch.sort(rank)[1]
        perm[idx_start:idx_end] = torch.as_tensor(idxs) + idx_start
    return perm


def fixed_order_loss(y, yhat, batch, loss_fn=cross_entropy_loss, order_fn=None):
    if order_fn is None:
        y_perm = slice(None)
    else:
        y_perm = fixed_order_idxs(y=y, batch=batch, order_fn=order_fn)
    y_ord = y[y_perm]
    # yhat_ord = yhat[y_perm]
    loss = torch.mean(loss_fn(yhat, y_ord))
    return loss