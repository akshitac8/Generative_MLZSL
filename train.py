#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:56:19 2020

@author: akshitac8
"""
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

########################################################
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

#calling the dataloader
data = util.DATA_LOADER(opt)
print("training samples: ", data.ntrain)

############## MODEL INITIALIZATION #############
netE = model.Encoder(opt)
netG = model.CLF(opt)
netD = model.Discriminator(opt)

print(netE)
print(netG)
print(netD)
################################################

#init tensors
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_test_labels = torch.LongTensor(opt.fake_batch_size, opt.nclass_all)
input_labels = torch.LongTensor(opt.batch_size, opt.nseen_class)
input_train_early_fusion_att = torch.FloatTensor(opt.batch_size, opt.attSize)
input_test_early_fusion_att = torch.FloatTensor(opt.fake_batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.attSize)
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netE.cuda()
    netG.cuda()
    netD.cuda()
    input_res = input_res.cuda()
    input_labels = input_labels.cuda()
    input_train_early_fusion_att = input_train_early_fusion_att.cuda()
    input_test_labels = input_test_labels.cuda()
    input_test_early_fusion_att = input_test_early_fusion_att.cuda()
    noise = noise.cuda()
    one = one.cuda()
    mone = mone.cuda()

def loss_fn(recon_x, x, mean, log_var):
    ## BCE+KL divergence loss
    BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(), size_average=False)
    BCE = BCE.sum() / x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
    return (BCE + KLD)

def sample():
    #train dataloader
    batch_labels, batch_feature, late_fusion_train_batch_att, early_fusion_train_batch_att = data.next_train_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_train_early_fusion_att.copy_(early_fusion_train_batch_att)
    input_labels.copy_(batch_labels)
    return late_fusion_train_batch_att

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

# setup optimizer
optimizerE = optim.Adam(netE.parameters(), lr=opt.lr)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

def calc_gradient_penalty(netD, real_data, fake_data, input_att=None):
    alpha = torch.rand(opt.batch_size, 1) 
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates.requires_grad = True
    if input_att is None:
        disc_interpolates = netD(interpolates)
    else:
        disc_interpolates = netD(interpolates, att=input_att)
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1)** 2).mean() * opt.lambda1
    return gradient_penalty


f1_best_GZSL_AP = 0
f1_best_GZSL_F1_5 = 0
f1_best_GZSL_F1_3 = 0
f1_best_ZSL_F1_5 = 0
f1_best_ZSL_F1_3 = 0

sum_f1_best_GZSL_F1 = 0
sum_f1_best_ZSL_F1 = 0

gzsl_best_epoch=0
zsl_best_epoch=0


tic1 = time.time()
#training loop
for epoch in range(0, opt.nepoch+1):
    mean_lossD = 0
    mean_lossG = 0
    mean_lossE = 0
    tic = time.time()
    for i in range(0, data.ntrain, opt.batch_size):
        ############################
        # (1) Update D network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in generator update

        for iter_d in range(opt.critic_iter):
            late_fusion_train_batch_att = sample()

            for param in netD.parameters():
                param.grad = None

            criticD_real = netD(input_res, att=input_train_early_fusion_att)
            criticD_real = opt.gammaD*criticD_real.mean()
            criticD_real.backward(mone)
        
            noise.normal_(0, 1)
            fake = netG(noise, att=late_fusion_train_batch_att, avg_att=input_train_early_fusion_att)
            criticD_fake = netD(fake.detach(), att=input_train_early_fusion_att)
            criticD_fake = opt.gammaD*criticD_fake.mean()
            criticD_fake.backward(one)

            gradient_penalty = opt.gammaD * calc_gradient_penalty(netD, input_res, fake.data, input_train_early_fusion_att)
            gradient_penalty.backward()
            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty
            optimizerD.step()
            mean_lossD += D_cost.item()

        ############################
        # (2) Update G network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters():
            p.requires_grad = False

        for param in netE.parameters():
            param.grad = None
        for param in netG.parameters():
            param.grad = None

        means, log_var = netE(input_res, att=input_train_early_fusion_att)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn([opt.batch_size, opt.attSize])
        if opt.cuda: eps=eps.cuda()
        z = eps * std + means

        recon_x = netG(z, att=late_fusion_train_batch_att, avg_att=input_train_early_fusion_att)
        vae_loss_seen = loss_fn(recon_x, input_res, means, log_var)
        mean_lossE += vae_loss_seen.item()
        errG = vae_loss_seen

        noise.normal_(0, 1)
        fake = netG(noise, att=late_fusion_train_batch_att, avg_att=input_train_early_fusion_att)
        criticG_fake = netD(fake, att=input_train_early_fusion_att).mean()
        G_cost = -criticG_fake
        errG += opt.gammaG*G_cost
        mean_lossG += G_cost.item()

        errG.backward()
        optimizerE.step()
        optimizerG.step()

    mean_lossG /= data.ntrain / opt.batch_size
    mean_lossD /= data.ntrain / opt.batch_size
    mean_lossE /= data.ntrain / opt.batch_size

    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Loss_E: %.4f, Wasserstein_dist: %.4f' %
            (epoch, opt.nepoch, mean_lossD, mean_lossG, mean_lossE, Wasserstein_D.item()))
    
    print("Generator {}th finished time taken {}".format(epoch, time.time()-tic))
    netG.eval()
    gzsl_syn_feature, gzsl_syn_label = generate_syn_feature(netG, data.GZSL_fake_test_labels, opt.fake_batch_size)
    if opt.gzsl:
        nclass = opt.nclass_all
        train_X = gzsl_syn_feature
        train_Y = gzsl_syn_label

        print(train_Y.shape)
        tic = time.time()
        gzsl_cls = classifier.CLASSIFIER(train_X, train_Y, data, nclass,
                                        opt.cuda, opt, opt.classifier_lr, 0.5, opt.classifier_epoch,
                                        opt.classifier_batch_size, True)

        sum_GZSL_F1_5 = gzsl_cls.sum_F1_scores_seen_unseen[4]*100 + gzsl_cls.sum_F1_scores_seen_unseen[0]*100
        if sum_f1_best_GZSL_F1 < sum_GZSL_F1_5:
            gzsl_best_epoch = epoch
            sum_f1_best_GZSL_F1 = sum_GZSL_F1_5
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
    
    ######### FETCHING ZSL CLASSIFIER TRAINING DATA ########################
    temp_label = gzsl_syn_label[:,:len(data.seenclasses)].sum(1)
    zsl_syn_label = gzsl_syn_label[temp_label==0][:,len(data.seenclasses):]
    zsl_syn_feature = gzsl_syn_feature[temp_label==0]
    print("ZSL DATA", zsl_syn_label.shape)
    ###############################################3########################


    tic = time.time()
    zsl_cls = classifier.CLASSIFIER(zsl_syn_feature, zsl_syn_label, data,
                                        data.unseenclasses.size(0), opt.cuda, opt, opt.classifier_lr,
                                        0.5, opt.classifier_epoch, opt.classifier_batch_size, False)

    sum_ZSL_F1 = zsl_cls.sum_F1_scores[4]*100 + zsl_cls.sum_F1_scores[0]*100
       
    if sum_f1_best_ZSL_F1 < sum_ZSL_F1:
        zsl_best_epoch = epoch
        sum_f1_best_ZSL_F1 = sum_ZSL_F1
        sum_f1_best_ZSL_AP = zsl_cls.sum_F1_scores[0]
        sum_f1_best_ZSL_F1_3 = zsl_cls.sum_F1_scores[1]
        sum_f1_best_ZSL_P_3 = zsl_cls.sum_F1_scores[2]
        sum_f1_best_ZSL_R_3 = zsl_cls.sum_F1_scores[3]
        sum_f1_best_ZSL_F1_5 = zsl_cls.sum_F1_scores[4]
        sum_f1_best_ZSL_P_5 = zsl_cls.sum_F1_scores[5]
        sum_f1_best_ZSL_R_5 = zsl_cls.sum_F1_scores[6]
       
    print('ZSL: AP=%.4f' % (zsl_cls.sum_F1_scores[0]))
    print('ZSL K=5 : f1=%.4f,P=%.4f,R=%.4f' % (zsl_cls.sum_F1_scores[4], zsl_cls.sum_F1_scores[5], zsl_cls.sum_F1_scores[6]))
    print('ZSL K=3 : f1=%.4f,P=%.4f,R=%.4f' % (zsl_cls.sum_F1_scores[1], zsl_cls.sum_F1_scores[2], zsl_cls.sum_F1_scores[3]))
    print("ZSL classification finished time taken {}".format(time.time()-tic))

    if epoch % 3 == 0 and epoch > 0: ## PRINT BEST EPOCH AFTER EVERY 3 EPOCHS
        print("LAST GZSL BEST EPOCH", gzsl_best_epoch)
        print('GZSL: AP=%.4f' % (sum_f1_best_GZSL_AP))
        print('GZSL K=5 : f1=%.4f,P=%.4f,R=%.4f' %
                (sum_f1_best_GZSL_F1_5, sum_f1_best_GZSL_P_5, sum_f1_best_GZSL_R_5))
        print('GZSL K=3 : f1=%.4f,P=%.4f,R=%.4f' %
                (sum_f1_best_GZSL_F1_3, sum_f1_best_GZSL_P_3, sum_f1_best_GZSL_R_3))
        print("LAST ZSL BEST EPOCH", zsl_best_epoch)
        print('ZSL: AP=%.4f' % (sum_f1_best_ZSL_AP))
        print('ZSL K=5 : f1=%.4f,P=%.4f,R=%.4f' %
                (sum_f1_best_ZSL_F1_5, sum_f1_best_ZSL_P_5, sum_f1_best_ZSL_R_5))
        print('ZSL K=3 : f1=%.4f,P=%.4f,R=%.4f' %
                (sum_f1_best_ZSL_F1_3, sum_f1_best_ZSL_P_3, sum_f1_best_ZSL_R_3))
    
    # reset G to training mode
    netG.train()

print(" Total time taken {} ".format(time.time()-tic1))

print("GZSL BEST EPOCH", gzsl_best_epoch)
print('GZSL: AP=%.4f' % (sum_f1_best_GZSL_AP))
print('GZSL K=5 : f1=%.4f,P=%.4f,R=%.4f' %
        (sum_f1_best_GZSL_F1_5, sum_f1_best_GZSL_P_5, sum_f1_best_GZSL_R_5))
print('GZSL K=3 : f1=%.4f,P=%.4f,R=%.4f' %
        (sum_f1_best_GZSL_F1_3, sum_f1_best_GZSL_P_3, sum_f1_best_GZSL_R_3))

print("ZSL BEST EPOCH", zsl_best_epoch)
print('ZSL: AP=%.4f' % (sum_f1_best_ZSL_AP))
print('ZSL K=5 : f1=%.4f,P=%.4f,R=%.4f' %
        (sum_f1_best_ZSL_F1_5, sum_f1_best_ZSL_P_5, sum_f1_best_ZSL_R_5))
print('ZSL K=3 : f1=%.4f,P=%.4f,R=%.4f' %
        (sum_f1_best_ZSL_F1_3, sum_f1_best_ZSL_P_3, sum_f1_best_ZSL_R_3))

##saving results to csv file
fname = 'CLF_result_F1.csv'
row = [opt.nepoch, sum_f1_best_GZSL_AP, sum_f1_best_ZSL_AP, sum_f1_best_GZSL_F1_3, sum_f1_best_GZSL_P_3, 
        sum_f1_best_GZSL_R_3, sum_f1_best_ZSL_F1_3, sum_f1_best_ZSL_P_3, sum_f1_best_ZSL_R_3,
        sum_f1_best_GZSL_F1_5, sum_f1_best_GZSL_P_5, sum_f1_best_GZSL_R_5, sum_f1_best_ZSL_F1_5, 
        sum_f1_best_ZSL_P_5, sum_f1_best_ZSL_R_5, opt.summary]

with open(fname, 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(row)
csvFile.close()
