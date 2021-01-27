import torch
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import os
import pickle
import h5py
import time
import numpy as np
import random
random.seed(3483)
np.random.seed(3483)

def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + '/')
    return ans

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# SYNTHESIS LABELS FROM TRAIN DATA 
def generate_fake_test_from_train_labels(train_seen_label, attribute, seenclasses, unseenclasses, num, per_seen=0.10, \
                                        per_unseen=0.40, per_seen_unseen= 0.50, dataset='NUS-WIDE'):
    if train_seen_label.min() == 0:
        print("Training data already trimmed and converted")
    else:
        print("original training data received (-1,1)'s ")
        train_seen_label = torch.clamp(train_seen_label,0,1)

    #remove all zero labeled images while training
    train_seen_label = train_seen_label[(train_seen_label.sum(1) != 0).nonzero().flatten()]
    seen_attributes = attribute[seenclasses]
    unseen_attributes = attribute[unseenclasses]
    if dataset=='NUS-WIDE':
        seen_percent, unseen_percent, seen_unseen_percent = per_seen , per_unseen, per_seen_unseen
    else:
        seen_percent, unseen_percent, seen_unseen_percent = per_seen , per_unseen, per_seen_unseen

    print("seen={}, unseen={}, seen-unseen={}".format(seen_percent, unseen_percent, seen_unseen_percent))
    print("syn num={}".format(num))
    gzsl = []
    for i in range(0, num):
        new_gzsl_syn_list = []
        seen_unseen_label_pairs = {}
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(unseen_attributes)
        for seen_idx, seen_att in zip(seenclasses,seen_attributes):
            _, indices = nbrs.kneighbors(seen_att[None,:])
            seen_unseen_label_pairs[seen_idx.tolist()] = unseenclasses[indices[0][0]].tolist()

        #ADDING ONLY SEEN LABELS
        idx = torch.randperm(len(train_seen_label))[0:int(len(train_seen_label)*seen_percent)]
        seen_labels = train_seen_label[idx]
        _new_gzsl_syn_list = torch.zeros(seen_labels.shape[0], attribute.shape[0])
        _new_gzsl_syn_list[:,:len(seenclasses)] = seen_labels
        new_gzsl_syn_list.append(_new_gzsl_syn_list)

        #ADDING ONLY UNSEEN LABELS
        idx = torch.randperm(len(train_seen_label))[0:int(len(train_seen_label)*unseen_percent)]
        temp_label = train_seen_label[idx]
        _new_gzsl_syn_list = torch.zeros(temp_label.shape[0], attribute.shape[0])
        for m,lab in enumerate(temp_label):
            new_lab = torch.zeros(attribute.shape[0])
            unseen_lab = lab.nonzero().flatten()
            u=[]
            for i in unseen_lab:
                u.append(seen_unseen_label_pairs[i.tolist()])
            new_lab[u]=1
            _new_gzsl_syn_list[m,:] = new_lab
        unseen_labels = _new_gzsl_syn_list
        new_gzsl_syn_list.append(unseen_labels)

        #ADDING BOTH SEEN AND UNSEEN LABELS 50% OF THE SELECTED SEEN LABELS IS MAPPED TO UNSEEN LABELS
        idx = torch.randperm(len(train_seen_label))[0:int(len(train_seen_label)*seen_unseen_percent)]
        temp_label = train_seen_label[idx]
        _new_gzsl_syn_list = torch.zeros(temp_label.shape[0], attribute.shape[0])
        for m,lab in enumerate(temp_label):
            u = []
            new_lab = torch.zeros(attribute.shape[0])
            seen_unseen_lab = lab.nonzero().flatten()
            temp_seen_label = np.random.choice(seen_unseen_lab,int(len(seen_unseen_lab)*0.50))
            u.extend(temp_seen_label)
            rem_seen_label =  np.setxor1d(temp_seen_label,seen_unseen_lab)
            for i in rem_seen_label:
                u.append(seen_unseen_label_pairs[i.tolist()])
            new_lab[u]=1
            _new_gzsl_syn_list[m,:] = new_lab
        seen_unseen_labels = _new_gzsl_syn_list
        new_gzsl_syn_list.append(seen_unseen_labels)

        new_gzsl_syn_list = torch.cat(new_gzsl_syn_list)
        gzsl.append(new_gzsl_syn_list)
    
    gzsl = torch.cat(gzsl)
    tmp_list = gzsl.sum(0)
    ## To make sure every unseen label gets covered
    empty_lab = torch.arange(tmp_list.numel())[tmp_list==0]
    min_uc = int(tmp_list[len(seenclasses):][tmp_list[len(seenclasses):]>0].min().item())
    for el in empty_lab:
        idx = torch.randperm(gzsl.size(0))[:min_uc]
        gzsl[idx,el] = 1
    gzsl = gzsl.long()
    print("GZSL TEST LABELS:",gzsl.shape)
    return gzsl

def get_seen_unseen_classes(file_tag1k, file_tag81):
    with open(file_tag1k, "r") as file:
        tag1k = np.array(file.read().splitlines())
    with open(file_tag81, "r") as file:
        tag81 = np.array(file.read().splitlines())
    seen_cls_idx = np.array(
        [i for i in range(len(tag1k)) if tag1k[i] not in tag81])
    unseen_cls_idx = np.array(
        [i for i in range(len(tag1k)) if tag1k[i] in tag81])
    return seen_cls_idx, unseen_cls_idx


class DATA_LOADER(object):
    def __init__(self, opt):
        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matdataset(self, opt):
        tic = time.time()
        src = "/home/ag1/akshita/multi-label-zsl/data"
        att_path = os.path.join(src,'wiki_contexts','NUS_WIDE_pretrained_w2v_glove-wiki-gigaword-300')
        file_tag1k = os.path.join(src,'NUS_WID_Tags','TagList1k.txt')
        file_tag81 = os.path.join(src,'ConceptsList','Concepts81.txt')
        self.seen_cls_idx, _ = get_seen_unseen_classes(file_tag1k, file_tag81)
        src_att = pickle.load(open(att_path, 'rb'))
        print("attributes are combined in this order-> seen+unseen")
        self.attribute = torch.from_numpy(normalize(np.concatenate((src_att[0][self.seen_cls_idx],src_att[1]),axis=0)))
        #VGG features path
        train_loc = load_dict_from_hdf5(os.path.join(src, 'nus_wide','nus_seen_train_vgg19.h5'))
        test_unseen_loc = load_dict_from_hdf5(os.path.join(src, 'nus_wide', 'nus_zsl_test_vgg19.h5'))
        test_seen_unseen_loc = load_dict_from_hdf5(os.path.join(src, 'nus_wide', 'nus_gzsl_test_vgg19.h5'))


        feature_train_loc = train_loc['features']
        label_train_loc = train_loc['labels']
        feature_test_unseen_loc = test_unseen_loc['features']
        label_test_unseen_loc = test_unseen_loc['labels']
        feature_test_seen_unseen_loc = test_seen_unseen_loc['features']
        label_test_seen_unseen_loc = test_seen_unseen_loc['labels']
        print("Data loading finished, Time taken: {}".format(time.time()-tic))

        tic = time.time()
        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature_train_loc)
                _test_unseen_feature = scaler.transform(feature_test_unseen_loc)
                _test_seen_unseen_feature = scaler.transform(feature_test_seen_unseen_loc)

                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1/mx)
                self.train_label = torch.from_numpy(label_train_loc).long()

                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_label = torch.from_numpy(label_test_unseen_loc).long()

                self.test_seen_unseen_feature = torch.from_numpy(_test_seen_unseen_feature).float()
                self.test_seen_unseen_feature.mul_(1/mx)
                self.test_seen_unseen_label = torch.from_numpy(label_test_seen_unseen_loc).long()
            else:
                self.train_feature = torch.from_numpy(feature_train_loc).float()
                self.train_label = torch.from_numpy(label_train_loc).long()
                self.test_unseen_feature = torch.from_numpy(feature_test_unseen_loc).float()
                self.test_unseen_label = torch.from_numpy(label_test_unseen_loc).long()

        print("REMOVING ZEROS LABELS")
        temp_label = torch.clamp(self.train_label,0,1)
        temp_seen_labels = temp_label.sum(1)
        temp_label = temp_label[temp_seen_labels>0]

        self.train_label           = temp_label
        self.train_feature         = self.train_feature[temp_seen_labels>0]

        self.train_trimmed_label   = self.train_label[temp_label.sum(1)<=opt.N]
        self.train_trimmed_feature = self.train_feature[temp_label.sum(1)<=opt.N]

        print("Data with N={} labels={}".format(opt.N,self.train_trimmed_label.shape))
        print("Full Data labels={} with min label/feature = {} and max label/feature = {}".format(self.train_label.shape, temp_label.sum(1).min(), temp_label.sum(1).max()))

        
        self.seenclasses = torch.from_numpy(np.arange(0, self.seen_cls_idx.shape[-1]))  # [0-925]
        self.unseenclasses = torch.from_numpy(np.arange(0+self.seen_cls_idx.shape[-1], len(self.attribute)))  # [925-1006]
        
        self.N = opt.N
        self.syn_num = opt.syn_num
        self.dataset = opt.dataset
        self.per_seen = opt.per_seen
        self.per_unseen = opt.per_unseen
        self.per_seen_unseen = opt.per_seen_unseen

        print("USING TRAIN FEATURES WITH <=N")
        self.ntrain = self.train_trimmed_feature.size()[0]
        train_labels = self.train_trimmed_label


        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()
        self.GZSL_fake_test_labels = generate_fake_test_from_train_labels(train_labels, self.attribute, self.seenclasses, \
                                        self.unseenclasses, self.syn_num, self.per_seen, self.per_unseen, self.per_seen_unseen, \
                                        dataset=opt.dataset)

        print("Data preprocssing finished, Time taken: {}".format(time.time()-tic))
    
    def _average(self, lab, attribute):
        return torch.mean(attribute[lab], 0)

    def early_fusion_preprocess_att(self, labels, attribute):
        new_seen_attribute = torch.zeros(labels.shape[0], attribute.shape[-1])
        for i in range(len(labels)):
            lab = labels[i].nonzero().flatten()
            if len(lab) == 0: continue
            new_seen_attribute[i, :] = self._average(lab, attribute)
        return new_seen_attribute

    def late_fusion_preprocess_att(self, labels, attribute):
        new_attributes = torch.zeros(labels.shape[0], self.N, attribute.shape[-1]) #new attributes [BS X 10 X 925]
        for i in range(len(labels)):
            lab = labels[i].nonzero().flatten()
            if len(lab) == self.N: new_attributes[i,:,:] = attribute[lab]
            elif len(lab) < self.N: new_attributes[i,:,:] = torch.cat((attribute[lab],torch.zeros((self.N - len(lab)), attribute.shape[-1])))
        return new_attributes

    ## Training Dataloader
    def next_train_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        feature = self.train_trimmed_feature
        labels = self.train_trimmed_label
        batch_feature = feature[idx]
        batch_labels = labels[idx]
        early_fusion_train_batch_att = self.early_fusion_preprocess_att(batch_labels, self.attribute)
        late_fusion_train_batch_att = self.late_fusion_preprocess_att(batch_labels, self.attribute)
        return batch_labels, batch_feature, late_fusion_train_batch_att, early_fusion_train_batch_att

    ## Testing Dataloader
    def next_test_batch(self, batch_size):
        idx = torch.randperm(len(self.GZSL_fake_test_labels))[0:batch_size]
        batch_labels = self.GZSL_fake_test_labels[idx]
        early_fusion_test_batch_att = self.early_fusion_preprocess_att(batch_labels, self.attribute)
        late_fusion_test_batch_att = self.late_fusion_preprocess_att(batch_labels, self.attribute)

        return batch_labels, late_fusion_test_batch_att, early_fusion_test_batch_att