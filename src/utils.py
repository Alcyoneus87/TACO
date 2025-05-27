import argparse
import sys
import torch
import random
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import wandb
import datetime


class EarlyStopping:
    def __init__(self, patience=7, delta=0, path='checkpoint.pt', type='acc'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.type = type

    def __call__(self, val_loss, model):

        if self.type == 'loss':
            score = -val_loss

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

        elif self.type == 'acc':
            score = -val_loss

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score > self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)

        self.val_loss_min = val_loss


class SupClusterConLoss(nn.Module):
    def __init__(self, device, temperature, base_temperature, beta, label_num, alpha, gamma):
        super(SupClusterConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.device = device
        self.beta = beta
        self.label_num = label_num
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, features, labels, feats_label, num_labels, mask, logits_mask, logits_1):
        feats_norm = F.normalize(features, dim=1)
        feats_label_norm = F.normalize(feats_label, dim=1)
        # if feats_label_norm is not None and mask is not None:
        #     raise ValueError('Cannot define both `labels` and `mask`')
        # elif feats_label_norm is None and mask is None:
        #     mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        # elif feats_label_norm is not None:
        #     feats_label_norm = feats_label_norm.contiguous().view(-1, 1)
        #     if feats_label_norm.shape[0] != batch_size:
        #         raise ValueError('Num of labels does not match num of features')
        #     mask = torch.eq(feats_label_norm, feats_label_norm.T).float().to(device)
        # else:
        #     mask = mask.float().to(device)

        lab_sim_emb = logits_1.T
        weights = torch.triu(F.cosine_similarity(lab_sim_emb.unsqueeze(1).expand(-1, num_labels, -1),
                                      lab_sim_emb.unsqueeze(0).expand(num_labels, -1, -1), dim=-1), diagonal=1)
        weights = weights / weights.sum()
        cos_mat = weights * F.cosine_similarity(feats_label_norm.unsqueeze(1).expand(-1, num_labels, -1),
                                      feats_label_norm.unsqueeze(0).expand(num_labels, -1, -1), dim=-1)
        L = self.gamma * torch.exp(cos_mat.sum())
        #
        # lab_sim_emb = lab_sim_emb.T
        # weights = torch.triu(F.cosine_similarity(lab_sim_emb.unsqueeze(1).expand(-1, num_labels, -1),
        #                                         lab_sim_emb.unsqueeze(0).expand(num_labels, -1, -1), dim=-1), diagonal=1)
        # cos_mat = weights * F.cosine_similarity(feats_label_norm.unsqueeze(1).expand(-1, num_labels, -1),
        #                               feats_label_norm.unsqueeze(0).expand(num_labels, -1, -1), dim=-1)

        anchor_dot_contrast = torch.div(
            torch.matmul(feats_norm, feats_norm.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # mask = mask.repeat(feats_norm, feats_norm)  # (1, 1)
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(batch_size * feats_norm).view(-1, 1).to(device),
        #     0
        # )
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / \
                            (mask.sum(1) + (mask.sum(1) == 0) * 1e7)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        L += self.alpha * loss.mean()

        return L, torch.exp(cos_mat.sum()), loss.mean()