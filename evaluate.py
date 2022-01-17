#author: akshitac8
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import time
import random
import os
import csv
import numpy as np
import warnings
import networks.CLF_model as model
import classifier as classifier
from config import opt
import util as util
warnings.filterwarnings('ignore')

#############################################
#setting up seeds
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
torch.set_default_tensor_type('torch.FloatTensor')
cudnn.benchmark = True  # For speed i.e, cudnn autotuner
########################################################

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#init dataloader
data = util.DATA_LOADER(opt)
print("training batches: ", data.ntrain)
# print("Datatset", opt.dataset)

############## MODEL INITIALIZATION #############
netG = model.HYBRID_FUSION_ATTENTION(opt)
print(netG)
################################################

#init tensors
input_test_labels = torch.LongTensor(opt.fake_batch_size, opt.nclass_all)
input_test_early_fusion_att = torch.FloatTensor(opt.fake_batch_size, opt.attSize)
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netG.cuda()
    input_test_labels = input_test_labels.cuda()
    input_test_early_fusion_att = input_test_early_fusion_att.cuda()
    one = one.cuda()
    mone = mone.cuda()


def fake_sample(batch_size):
    #fake data synthesis dataloader
    batch_test_labels, late_fusion_test_batch_att, early_fusion_test_batch_att = data.next_test_batch(batch_size)
    input_test_labels.copy_(batch_test_labels)
    input_test_early_fusion_att.copy_(early_fusion_test_batch_att)
    return late_fusion_test_batch_att


def generate_syn_feature(netG, classes, batch_size):
    ## SYNTHESIS MULTI LABEL FEATURES
    nsample = classes.shape[0]  # zsl_classes or gzsl_classes
    if not nsample % batch_size == 0:
        nsample = nsample + (batch_size - (nsample % batch_size))
    nclass = classes.shape[1]
    syn_noise = torch.FloatTensor(batch_size, opt.attSize)
    syn_feature = torch.FloatTensor(nsample, opt.resSize)
    syn_label = torch.LongTensor(nsample, classes.shape[1])
    if opt.cuda:
        syn_noise = syn_noise.cuda()
    for k, i in enumerate(range(0, nsample, batch_size)):
        late_fusion_test_batch_att = fake_sample(batch_size)
        syn_noise.normal_(0, 1)
        with torch.no_grad():
        output = netG(syn_noise, att=late_fusion_test_batch_att, avg_att=input_test_early_fusion_att)
        syn_feature.narrow(0, k*batch_size, batch_size).copy_(output)
        syn_label.narrow(0, k*batch_size, batch_size).copy_(input_test_labels)
    return syn_feature, syn_label


save_path = "weights/best_ZSL_model.pth.tar" ##saved model weights path
netG = netG.cuda()
netG.load_state_dict(torch.load(save_path))
print("classifier training")
netG.eval()
gzsl_syn_feature, gzsl_syn_label = generate_syn_feature(netG, data.GZSL_fake_test_labels, opt.fake_batch_size)

nclass = opt.nclass_all
train_X = gzsl_syn_feature
train_Y = gzsl_syn_label

print(train_Y.shape)
tic = time.time()
gzsl_cls = classifier.CLASSIFIER(train_X, train_Y, data, nclass,
                                opt.cuda, opt, opt.classifier_lr, 0.5, opt.classifier_epoch,
                                opt.classifier_batch_size, True)


sum_f1_best_GZSL_AP = gzsl_cls.sum_F1_scores_seen_unseen[0]
sum_f1_best_GZSL_F1_3 = gzsl_cls.sum_F1_scores_seen_unseen[1]
sum_f1_best_GZSL_P_3 = gzsl_cls.sum_F1_scores_seen_unseen[2]
sum_f1_best_GZSL_R_3 = gzsl_cls.sum_F1_scores_seen_unseen[3]
sum_f1_best_GZSL_F1_5 = gzsl_cls.sum_F1_scores_seen_unseen[4]
sum_f1_best_GZSL_P_5 = gzsl_cls.sum_F1_scores_seen_unseen[5]
sum_f1_best_GZSL_R_5 = gzsl_cls.sum_F1_scores_seen_unseen[6]
    
print('GZSL: AP=%.4f' % (gzsl_cls.sum_F1_scores_seen_unseen[0]))
print('GZSL K=5 : f1=%.4f,P=%.4f,R=%.4f' % (
    gzsl_cls.sum_F1_scores_seen_unseen[4], gzsl_cls.sum_F1_scores_seen_unseen[5], gzsl_cls.sum_F1_scores_seen_unseen[6]))
print('GZSL K=3 : f1=%.4f,P=%.4f,R=%.4f' % (
    gzsl_cls.sum_F1_scores_seen_unseen[1], gzsl_cls.sum_F1_scores_seen_unseen[2], gzsl_cls.sum_F1_scores_seen_unseen[3]))

print("GZSL classification finished time taken {}".format(time.time()-tic))

temp_label = gzsl_syn_label[:, :len(data.seenclasses)].sum(1)
zsl_syn_label = gzsl_syn_label[temp_label == 0][:, len(data.seenclasses):]
zsl_syn_feature = gzsl_syn_feature[temp_label == 0]

print(zsl_syn_label.shape)

tic = time.time()
zsl_cls = classifier.CLASSIFIER(zsl_syn_feature, zsl_syn_label, data,
                                data.unseenclasses.size(0), opt.cuda, opt, opt.classifier_lr,
                                0.5, opt.classifier_epoch, opt.classifier_batch_size, False)
                                
sum_f1_best_ZSL_AP = zsl_cls.sum_F1_scores[0]
sum_f1_best_ZSL_F1_3 = zsl_cls.sum_F1_scores[1]
sum_f1_best_ZSL_P_3 = zsl_cls.sum_F1_scores[2]
sum_f1_best_ZSL_R_3 = zsl_cls.sum_F1_scores[3]
sum_f1_best_ZSL_F1_5 = zsl_cls.sum_F1_scores[4]
sum_f1_best_ZSL_P_5 = zsl_cls.sum_F1_scores[5]
sum_f1_best_ZSL_R_5 = zsl_cls.sum_F1_scores[6]

print("ZSL classification finished time taken {}".format(time.time()-tic))

print('ZSL: AP=%.4f' % (zsl_cls.sum_F1_scores[0]))
print('ZSL K=5 : f1=%.4f,P=%.4f,R=%.4f' % (zsl_cls.sum_F1_scores[4], zsl_cls.sum_F1_scores[5], zsl_cls.sum_F1_scores[6]))
print('ZSL K=3 : f1=%.4f,P=%.4f,R=%.4f' % (zsl_cls.sum_F1_scores[1], zsl_cls.sum_F1_scores[2], zsl_cls.sum_F1_scores[3]))
