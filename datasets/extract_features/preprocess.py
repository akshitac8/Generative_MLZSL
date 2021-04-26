#author: akshitac8
import os
from glob import glob
import json
import numpy as np
from time import time
import shutil
import torch
import argparse
np.random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--train_json', type=str, default='nus_wide_train', help='training json filename')
parser.add_argument('--test_json', type=str, default='nus_wide_test', help='testing json filename')
parser.add_argument('--image_dir', type=str, default='Flicker', help='foldername containg all the images')
parser.add_argument('--output_dir', type=str, default='Flicker_jsons', help='foldername containg generated jsons')

def load_labels_81(filename, tag81):
    """
    Input:
        filename -> folder containing tag files of 81 categories which contains image level annotations
        tag81 -> NUS-WIDE provided Taglist of 81 categories.
    Output:
        label_tags -> 

    """
    label_tags = []
    for tag in tag81:
        with open(filename+'Labels_{}.txt'.format(tag), "r") as file:
            label_tag = np.array(file.read().splitlines(),dtype=np.float32)[:, np.newaxis]
            label_tags.append(label_tag)
    label_tags = np.concatenate(label_tags, axis=1)
    return label_tags

def get_seen_unseen_classes(file_tag1k, file_tag81):
   """
    Input:
        file_tag1k -> NUS-WIDE provided Taglist of 1000 categories.
        file_tag81 -> NUS-WIDE provided Taglist of 81 categories.
        
    Output:
        seen_cls_idx -> selected seen class indices
        unseen_cls_idx -> selected unseen class indices
    """
    with open(file_tag1k, "r") as file:
        tag1k = np.array(file.read().splitlines())
    with open(file_tag81, "r") as file:
        tag81 = np.array(file.read().splitlines())
    seen_cls_idx = np.array([i for i in range(len(tag1k)) if tag1k[i] not in tag81])
    unseen_cls_idx = np.array([i for i in range(len(tag1k)) if tag1k[i] in tag81])
    return seen_cls_idx, unseen_cls_idx


def load_id_label_imgs(id_filename, data_partition, label1k_filename, label81_human_filename, tag81):
    """
    Input:
        id_filename ->
        data_partition ->
        label1k_filename ->
        label81_human_filename ->
        tag81 ->
    Output:
        dict_img_id ->
        idxs_partition ->
        label1k_imgs ->
        label81_imgs ->

    """
    with open(id_filename, "r") as file:
        id_imgs = file.readlines()
        id_imgs = [id_img.rstrip().replace('\\', '/') for id_img in id_imgs]

    with open(data_partition, "r") as file:
        idxs_partition = file.readlines()
        idxs_partition = [idx.rstrip().replace('\\', '/') for idx in idxs_partition]

    with open(label1k_filename, "r") as file:
        label1k_imgs = file.readlines()

    dict_img_id = {}
    for idx, id_img in enumerate(id_imgs):
        key = id_img.split('/')[-1]
        dict_img_id[key] = idx

    label1k_imgs = np.float32([label_img.strip().split() for label_img in label1k_imgs])
    label81_imgs = load_labels_81(label81_human_filename, tag81)
    return dict_img_id, idxs_partition, label1k_imgs, label81_imgs

def get_labels(img_id, dict_img_id, label81_imgs, label1k_imgs):
    """
    Input:
        img_id ->
        dict_img_id ->
        label81_imgs ->
        label1k_imgs ->
    Output:
        label_tags ->

    """
    idx_dict = dict_img_id[img_id]
    # The result is different between AllTags81 and labels_{}. AllTag81 is collected from flicker Tags whereas labels_{} is annotated by human.
    label81 = 2*label81_imgs[idx_dict]-1
    label1k = 2*label1k_imgs[idx_dict]-1
    label1k[unseen_cls_idx] = 0
    return label81, label1k

data_set = ['Train', 'Test']
file_tag1k = 'NUS_WID_Tags/TagList1k.txt'
file_tag81 = 'ConceptsList/Concepts81.txt'
seen_cls_idx, unseen_cls_idx = get_seen_unseen_classes(file_tag1k, file_tag81)
id_filename = 'ImageList/Imagelist.txt'
label1k_filename = 'NUS_WID_Tags/AllTags1k.txt'
label81_human_filename = 'AllLabels/'
src_image = opt.image_dir

for _data in data_set:
    print('data_set {}'.format(_data))
    data_partition = 'ImageList/{}Imagelist.txt'.format(_data)
    dict_img_id, idxs_partition, label1k_imgs, label81_imgs = load_id_label_imgs(id_filename, data_partition, label1k_filename, label81_human_filename, tag81)
    src = opt.output_dir
    if _data == 'Train':
        train_seen_json = os.path.join(src, opt.train_json+'.json')
        _img_json = {}
        for index, img_id in enumerate(idxs_partition):
            key = img_id.split('/')[-1]
            label81, _ = get_labels(key, dict_img_id, label81_imgs, label1k_imgs)
            seen_annotations = label_1k[seen_cls_idx]

            _img_json[img_id] = {}
            _img_json[img_id]['labels_925'] = seen_annotations.tolist()
            _img_json[img_id]['labels_81'] = label81.tolist()

        with open(train_seen_json,'w') as q:
            json.dump(_img_json,q)

    elif _data == 'Test':
        test_json = os.path.join(src, opt.test_json+'.json')
        img_json = {}
        for index, img_id in enumerate(idxs_partition):
            key = img_id.split('/')[-1]
            label_81, _ = get_labels(key, dict_img_id, label81_imgs, label1k_imgs)
            su_annotations = np.concatenate((label_1k[seen_cls_idx],label_81))
            unseen_annotations = labels_81

            img_json[img_id] = {}
            img_json[img_id]['labels_81'] = unseen_annotations.tolist()
            img_json[img_id]['labels_1006'] = su_annotations.tolist()

        print(len(img_json))
        with open(test_json,'w') as q:
            json.dump(img_json,q)