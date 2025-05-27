import argparse
import sys, os
import torch
import random
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import wandb
import datetime
from datasets import *
from models import CrossEncoder
from utils import EarlyStopping
# import plotly.express as px
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings("ignore")


def train(model, train_loader, opt, schedule):
    model.train()

    total_loss, all_preds, all_labels = 0, [], []
    cnt_1, cnt_2 = 0, 0
    for input_ids, attention_mask, labels in tqdm(train_loader):
        input_ids, attention_mask, labels = input_ids.to(args.device), attention_mask.to(args.device), \
            labels.to(args.device)
        logits, loss = model(input_ids, attention_mask, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()
        schedule.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=-1).tolist())
        all_labels.extend(labels.tolist())

    train_acc = accuracy_score(all_labels, all_preds)
    train_weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    marco_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'Train => loss : {total_loss / len(train_loader):.04f}, '
          f'acc : {train_acc:.04f}, '
          f'f1 : {train_weighted_f1:.04f}, '
          f'marco_f1 : {marco_f1:.04f}')

    return train_acc, train_weighted_f1, marco_f1


def eval_or_test(model, loader, mode):
    model.eval()

    all_logits = []
    total_loss,all_preds, all_labels = 0, [], []
    cnt_1, cnt_2 = 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids, attention_mask, labels = input_ids.to(args.device), attention_mask.to(args.device), \
                labels.to(args.device)
            logits, loss = model(input_ids, attention_mask, labels)

            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=-1).tolist())
            all_labels.extend(labels.tolist())
            all_logits.extend(logits.tolist())

    acc = accuracy_score(all_labels, all_preds)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    marco_f1 = f1_score(all_labels, all_preds, average='macro')
    print(mode + f' => loss : {total_loss / len(loader):.04f}, '
                 f'acc : {acc:.04f}, '
                 f'weighted_f1 : {weighted_f1:.04f}, '
                 f'marco_f1 : {marco_f1:.04f}')

    # if mode == 'test':
    #     print(f1_score(all_labels, all_preds))
    # if mode == 'Test':  en(all_logits)

    return acc, weighted_f1, marco_f1


def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def main():
    # args.lr = float("{:.5e}".format(wandb.config.lr))
    # args.epochs = wandb.config.epochs
    # # args.weight_decay = float("{:.4e}".format(wandb.config.weight_decay))
    # # args.beta = wandb.config.beta
    # args.temperature = float("{:.5e}".format(wandb.config.temperature))
    # args.gamma = float("{:.5e}".format(wandb.config.gamma))

    label2idx, idx2label = get_dicts(args.dataset)
    num_classes = len(idx2label.items())
    class_names = [v for k, v in sorted(idx2label.items(), key=lambda item: item[0])]

    tkr = AutoTokenizer.from_pretrained(args.encoder_type)
    special_tokens_dict = {'additional_special_tokens': ["<#>"]}
    tkr.add_special_tokens(special_tokens_dict)

    trainset = emotion_DataSet('train', tkr)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=trainset.collate)
    validset = emotion_DataSet('valid', tkr)
    valid_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False, collate_fn=trainset.collate)
    testset = emotion_DataSet('test', tkr)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, collate_fn=trainset.collate)

    model = CrossEncoder(args, num_classes, tkr)
    model.to(args.device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # , no_deprecation_warning=True
    schedule = get_linear_schedule_with_warmup(opt, num_warmup_steps=0,  # len(train_loader) * 3
                                               num_training_steps=len(train_loader) * args.epochs)
    early_stopping = EarlyStopping(patience=args.patience, type='acc', path='../mlp' + args.device_id +
                                                                            args.dataset + '.pt')

    for epoch in range(args.epochs):
        print(f'----------- {epoch + 1} -------------')

        if epoch + 1 > 0:   # :
            model.is_W_loss = True

        train_acc, train_f1, train_marco_f1 = train(model, train_loader, opt, schedule)
        eval_acc, eval_f1, eval_marco_f1 = eval_or_test(model, valid_loader, 'Eval')
        # wandb.log({
        #     'train_ce_acc': train_acc,
        #     'train_ce_f1': train_f1,
        #     'train_marco_f1': train_marco_f1,
        #     'eval_ce_acc': eval_acc,
        #     'eval_ce_f1': eval_f1,
        #     'eval_marco_f1': eval_marco_f1
        # })

        early_stopping(eval_acc + eval_f1, model)
        if early_stopping.early_stop:
            print(f"+++ early stop at epoch {epoch + 1 - args.patience} +++")
            break

    model.load_state_dict(torch.load('../mlp' + args.device_id + args.dataset + '.pt', map_location=args.device))
    print(f'----------- test -------------')
    test_acc, test_f1, test_marco_f1 = eval_or_test(model, test_loader, 'Test')
    print('test_acc_f1: ', test_acc + test_f1)
    # wandb.log({
    #     'test_acc': test_acc,
    #     'test_f1': test_f1,
    #     'test_marco_f1': test_marco_f1,
    # })

    # postfix = str(args.lr) + '_' + str(args.ce_lr) + '_' + str(args.temperature) \
    #           + '_' + str(args.scl_epochs) + '_' + str(args.ce_epochs)
    # postfix = '\_cluster'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ED')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--encoder_type', type=str, default='../roberta-base')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-05)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=4)
    # parser.add_argument('--beta', type=float, default=0.7)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.4)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--base_temperature', type=float, default=0.07)
    parser.add_argument('--device_id', type=str, default="0")

    args = parser.parse_args()
    args.device = torch.device("cuda:" + args.device_id if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    main()

    # sweep_configuration = {
    #     'method': 'bayes',
    # }
    # wandb.agent(sweep_id, function=main)