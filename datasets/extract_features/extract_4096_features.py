#author: akshitac8
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from VGG_model import Net
import json
from tqdm import tqdm
import numpy as np
import h5py
import argparse

np.random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--train_json', type=str, default='nus_wide_train', help='training json filename')
parser.add_argument('--test_json', type=str, default='nus_wide_test', help='testing json filename')
parser.add_argument('--output_dir', type=str, default='Flicker_jsons', help='foldername containg generated jsons')

def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes,'<U44')):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))


model = Net()
model = model.eval()
print(model)

GPU = True
if GPU:
    gpus = '0'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
      print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

    if len(device_ids)>1:
        model = nn.DataParallel(model, device_ids = device_ids).cuda()
    else:
        model = model.cuda()

jsons = [opt.train_json, opt.test_json]

for json_ in jsons: #type_ , types):
    type_ = 'Flickr'
    feat, features, labels, file_ = {}, [], [], []
    # import pdb;pdb.set_trace()
    dataset_ = get_extract_data(
        dir_ = os.path.join('/home/fk1/akshita/multi-label-zsl','data/{}'.format(type_)),
        json_file = os.path.join(opt.output_dir, json_+'.json'))  
    data_loader = DataLoader(dataset=dataset_, batch_size=64, shuffle=False, num_workers=16, drop_last=False)

    for data_ in tqdm(data_loader):
        filename, img, lab = data_[0], data_[1], data_[2]
        bs = img.size(0)
        if GPU:
            img = img.cuda()
        with torch.no_grad():
            out = model(img)
        for i in range(bs):
            features.append(out[i].detach().cpu().numpy().tolist())
            labels.append(lab[i].tolist())
            file_.append(filename[i].encode("ascii", "ignore"))

    print(np.array(features).astype('float32').min())
    print(np.array(features).astype('float32').max())

    print(np.array(features).astype('float32').mean())
    print(np.array(features).astype('float32').shape)
    feat['image_files'] = np.array(file_)#.astype(str)
    feat['features'] = np.array(features).astype('float32')
    feat['labels'] = np.array(labels).astype('float32')
    filename = json_+'_second_cls_layer_img_names_no_centercrop_relu_transform'
    save_dict_to_hdf5(feat, os.path.join(opt.output_dir, filename+'.h5'))
