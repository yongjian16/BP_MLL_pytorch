# -*- coding: utf-8 -*-

import torch

def pairwise_sub(a,b):
    column = a.unsqueeze(2)
    row = b.unsqueeze(1)
    return torch.sub(column,row)

def pairwise_and(a,b):
    column = a.unsqueeze(2)
    row = b.unsqueeze(1)
    return torch.mul(column,row)

def bp_mll_grad(y_pred,y_true):
    y_i = torch.eq(y_true,1)
    y_i_bar = torch.ne(y_true,1)
    y_i_sig = - y_true.clone()
    y_i_sig[y_i_sig==0] = 1

    truth_matrix = pairwise_and(y_i,y_i_bar).float()
    truth_matrix_bar = pairwise_and(y_i_bar,y_i).float()
    sub_matrix = pairwise_sub(y_pred,y_pred)
    exp_matrix = torch.exp(5*torch.mul(truth_matrix_bar-truth_matrix,sub_matrix))
    sparse_matrix = torch.mul(truth_matrix-truth_matrix_bar,exp_matrix)

    y_i_sizes = torch.sum(y_i.float(),1)
    y_i_bar_sizes = torch.sum(y_i_bar.float(),1)
    normalizers = torch.mul(y_i_sizes,y_i_bar_sizes) + 1e-6
    normalizers = normalizers.unsqueeze(1)
    normalizers = normalizers.unsqueeze(2)
    
    sig_matrix = torch.div(sparse_matrix,normalizers)
    grad = torch.sum(sig_matrix,1)
    return grad


def bp_mll_loss(y_pred,y_true):
    y_i = torch.eq(y_true,1)
    y_i_bar = torch.ne(y_true,1)

    truth_matrix = pairwise_and(y_i,y_i_bar).float()
    sub_matrix = pairwise_sub(y_pred,y_pred)

    exp_matrix = torch.exp(-5*sub_matrix)
    sparse_matrix = torch.mul(exp_matrix,truth_matrix)
    sums = torch.sum(sparse_matrix,[1,2])

    y_i_sizes = torch.sum(y_i.float(),1)
    y_i_bar_sizes = torch.sum(y_i_bar.float(),1)
    normalizers = torch.mul(y_i_sizes,y_i_bar_sizes) + 1e-6
    results = torch.div(sums,normalizers)

    return torch.mean(results)