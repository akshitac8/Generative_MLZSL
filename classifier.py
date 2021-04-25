#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:56:19 2020

@author: akshitac8
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import util as util
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import sys
import copy
import json
import csv
import os
import torch.nn.functional as F
from config import opt
import CLF_model as model
import random
import torch.backends.cudnn as cudnn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


class CLASSIFIER:
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, _cuda, opt, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, generalized=True):
        self.train_X = _train_X
        self.train_Y = _train_Y
        self.train_feature = data_loader.train_feature
        self.train_label = data_loader.train_label
        self.test_seen_unseen_feature = data_loader.test_seen_unseen_feature
        self.test_seen_unseen_label = data_loader.test_seen_unseen_label
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.size(1)
        self.cuda = _cuda
        self.model = LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
        self.model.apply(util.weights_init)
        self.input = torch.FloatTensor(_batch_size, self.input_dim)
        self.label = torch.LongTensor(_batch_size, self.nclass)
        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))
        if self.cuda:
            self.model.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()
            self.test_seen_unseen_feature = self.test_seen_unseen_feature.cuda()
            self.test_seen_unseen_label = self.test_seen_unseen_label.cuda()
            self.test_unseen_feature = self.test_unseen_feature.cuda()
            self.test_unseen_label = self.test_unseen_label.cuda()
            
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        if generalized:
            self.sum_F1_scores_seen_unseen, self.sum_f1_best_model =  self.fit()
        else:
            self.sum_F1_scores, self.sum_f1_best_model = self.fit_zsl()

    def fit(self):
        best_f1 = 0
        sum_f1_ap=0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                output = self.model(self.input)
                loss = F.binary_cross_entropy_with_logits(output, self.label.float()).cuda()
                loss.backward()
                self.optimizer.step()
            scores_su = self.val(self.test_seen_unseen_feature, self.test_seen_unseen_label)
            sum_ = scores_su[4]*100 + scores_su[0]*100
            if sum_ > sum_f1_ap:
                sum_f1_ap = sum_
                sum_F1_scores_seen_unseen = scores_su
                sum_f1_best_model = copy.deepcopy(self.model)
        return sum_F1_scores_seen_unseen, sum_f1_best_model

    def fit_zsl(self):
        best_f1 = 0
        sum_f1_ap = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                output = self.model(self.input)
                loss = F.binary_cross_entropy_with_logits(output, self.label.float()).cuda()
                loss.backward()
                self.optimizer.step()
            _scores = self.val(self.test_unseen_feature, self.test_unseen_label)
            sum_ = _scores[4]*100 + _scores[0]*100
            if sum_ > sum_f1_ap:
                sum_f1_ap = sum_
                sum_F1_scores = _scores
                sum_f1_best_model = copy.deepcopy(self.model)
        return sum_F1_scores, sum_f1_best_model
            
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0), torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]

    def val(self, test_X, test_label):
        start = 0
        ntest = test_X.size()[0]
        outputs = []
        labels = []
        for i in range(0, ntest, self.batch_size):            
            end = min(ntest, start+self.batch_size)
            with torch.no_grad():
                if self.cuda:
                    output = self.model(test_X[start:end].cuda())
                else:
                    output = self.model(test_X[start:end])
            label = test_label[start:end]
            outputs.extend(output)
            labels.extend(label)
            start = end
        outputs = torch.stack(outputs)
        labels = torch.stack(labels)
        outputs_ap = outputs.clone()
        outputs_3 = outputs.clone()
        outputs_5 = outputs.clone()
        ap = self.compute_AP(outputs_ap, labels)
        f1_3, p_3, r_3 = self.compute_F1(outputs_3, labels, 'overall', k_val=3)
        f1_5, p_5, r_5 = self.compute_F1(outputs_5, labels, 'overall', k_val=5)
        scores = (torch.mean(ap), torch.mean(f1_3), torch.mean(p_3), torch.mean(r_3), torch.mean(f1_5), torch.mean(p_5), torch.mean(r_5))
        return scores

    def compute_AP(self, predictions, labels):
        ## cuda ap computation
        num_class = predictions.size(1)
        ap = torch.zeros(num_class).cuda()
        for idx_cls in range(num_class):
            prediction = predictions[:, idx_cls]
            label = labels[:, idx_cls]
            mask = label.abs() == 1
            if (label > 0).sum() == 0:
                continue
            binary_label = torch.clamp(label[mask], min=0, max=1)
            sorted_pred, sort_idx = prediction[mask].sort(descending=True)
            sorted_label = binary_label[sort_idx]
            tmp = (sorted_label==1).float()
            tp = tmp.cumsum(0)
            fp = (sorted_label != 1).float().cumsum(0)
            num_pos = binary_label.sum()
            rec = tp/num_pos
            prec = tp/(tp+fp)
            ap_cls = (tmp*prec).sum()/num_pos
            ap[idx_cls].copy_(ap_cls)
        return ap

    def compute_F1(self, predictions, labels, mode_F1, k_val):
        ## cuda F1 computation
        idx = predictions.topk(dim=1, k=k_val)[1] 
        predictions.fill_(0)
        predictions.scatter_(dim=1,index=idx,src=torch.ones(predictions.size(0),k_val).cuda())
        if mode_F1 == 'overall':
            # print('evaluation overall!! cannot decompose into classes F1 score')
            mask = predictions == 1
            TP = (labels[mask] == 1).sum().float()
            tpfp = mask.sum().float()
            tpfn = (labels == 1).sum().float()
            p = TP/ tpfp
            r = TP/tpfn
            f1 = 2*p*r/(p+r)
        else:
            num_class = predictions.shape[1]
            # print('evaluation per classes')
            f1 = np.zeros(num_class)
            p = np.zeros(num_class)
            r = np.zeros(num_class)
            for idx_cls in range(num_class):
                prediction = np.squeeze(predictions[:, idx_cls])
                label = np.squeeze(labels[:, idx_cls])
                if np.sum(label > 0) == 0:
                    continue
                binary_label = np.clip(label, 0, 1)
                f1[idx_cls] = f1_score(binary_label, prediction)
                p[idx_cls] = precision_score(binary_label, prediction)
                r[idx_cls] = recall_score(binary_label, prediction)
        return f1, p, r


class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc1 = nn.Linear(input_dim, nclass)

    def forward(self, x):
        o = self.fc1(x)
        return o
