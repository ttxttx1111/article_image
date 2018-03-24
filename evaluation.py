import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
# from torch.nn.utils.rnn import pack_padded_sequence
from model import ImageCNN, MatchCNN

import argparse
import os
import pickle
from data_loader import get_loader, CocoDataset
from build_vocab import Vocabulary
from torchvision import transforms
import time
from pycocotools.coco import COCO
from PIL import Image
import nltk
from random import shuffle
from matchCNN_st import MatchCNN_st


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


"""load coco dataset"""
data_dir = "../data/coco/"
annotation_file = data_dir + "annotations/captions_val2014.json"
coco = COCO(annotation_file)

# anns = coco.anns
# imgs = coco.imgs

"""extract 200 imgid and corresponding 1000 captionid"""
sample_num = 1000
# caption_num = sample_num * 1000
img_ids_all = list(coco.imgs.keys())
shuffle(img_ids_all)
img_ids = []
ann_ids = []

for key in img_ids_all:
    temp = coco.getAnnIds(key)
    ann_ids.append(temp[0])
    img_ids.append(key)
    if len(img_ids) == sample_num:
        break

"""preprocess images"""
image_dir = data_dir + "resized2014/"
imgs = []

# Image preprocessing
transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

"""
imgs:
    list of img
img:
    ann_ids, data, id
"""
for i, img_id in enumerate(img_ids):
    img_new = {}
    img = coco.imgs[img_id]
    image = Image.open(image_dir + img["file_name"]).convert("RGB")
    image = transform(image)
    img_new["ann_ids"] = ann_ids[i]
    img_new["data"] = image
    img_new["id"] = img_id
    imgs.append(img_new)

"""preprocess annotations"""
vocab_file = "../data/coco/vocab.pkl"
pad_len = 62
# Load vocabulary wrapper.
with open(vocab_file, 'rb') as f:
    vocab = pickle.load(f)

anns = np.zeros((sample_num, pad_len), dtype=int)
for i, ann_id in enumerate(ann_ids):
    # for j, ann_id in enumerate(ann_ids_image):
        caption_str = coco.anns[ann_id]["caption"]
        tokens = nltk.tokenize.word_tokenize(str(caption_str).lower())
        caption = []
        #         caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        #         caption.append(vocab('<end>'))
        caption = np.array(caption)
        anns[i][:len(tokens)] = caption[:]

anns = to_var(torch.from_numpy(anns))

"""parameters"""
image_vector_size = 256
embed_size = 100
margin = 0.1
batch_size = 10
vocab_size = 9956
momentum = 0.9
lr = 0.0001
pad_len = 62
num_workers = 2

"""set model"""
imageCNN = ImageCNN(image_vector_size=image_vector_size)
matchCNN = MatchCNN_st(embed_size=embed_size,
                       image_vector_size=image_vector_size,
                       vocab_size=vocab_size,
                       pad_len=pad_len)

if torch.cuda.is_available():
    print("cuda is available")
    imageCNN = imageCNN.cuda()
    matchCNN = matchCNN.cuda()



"""load models"""
model_path = "../models"
imageCNN.load_state_dict(torch.load(os.path.join(model_path, 'imageCNN_mar0.5_st39-0.005489.pkl')))
matchCNN.load_state_dict(torch.load(os.path.join(model_path, 'matchCNN_mar0.5_st39-0.005489.pkl')))

# imageCNN = imageCNN.eval()
# matchCNN = matchCNN.eval()

"""extract image feature"""

img_data = to_var(torch.zeros(sample_num, 3, 224, 224))
img_features_batch = to_var(torch.zeros(sample_num // batch_size, batch_size, image_vector_size))
for i, img in enumerate(imgs):
    img_data[i] = img["data"]

# img_data_batch = to_var(torch.zeros(sample_num // batch_size, batch_size, 3, 224, 224))
for i in range(sample_num // batch_size):
    img_features_batch[i] = imageCNN(img_data[i * batch_size:(i + 1) * batch_size])

scores = np.zeros((sample_num, sample_num))
for i, caption in enumerate(anns):
    caption_tmp = caption.unsqueeze(0)
    caption_batch = caption_tmp.repeat(batch_size, 1)
    for j, img_feature_batch in enumerate(img_features_batch):
        score_batch = matchCNN(img_feature_batch, caption_batch)
        score_batch_np = score_batch.cpu().data.numpy()

        scores[j * batch_size:(j + 1) * batch_size, i] = score_batch_np[:, 0]


# sort by column
sorted_scores = (-scores).argsort(axis=0)

scores_ranks = np.zeros((sample_num, sample_num), dtype=int)

for i in range(sample_num):
    for j in range(sample_num):
        scores_ranks[sorted_scores[i][j]][j] = i

ranks_image = np.zeros((sample_num), dtype=int)
for i in range(sample_num):
    ranks_image[i] = scores_ranks[i][i]

# sorted_ranks_image = np.sort(ranks_image)
# med_ranks = np.zeros(sample_num)
# for i in range(sample_num):
#     med_ranks[i] = sorted_ranks_image[i][0]

r1 = len(ranks_image[ranks_image == 0]) / sample_num * 100
r5 = len(ranks_image[ranks_image <= 4]) / sample_num * 100
r10 = len(ranks_image[ranks_image <= 9]) / sample_num * 100
med = np.mean(ranks_image)

print("r1:", r1)
print("r5:", r5)
print("r10:", r10)
print("med:", med)
print("")
