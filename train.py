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
from data_loader import get_loader
from build_vocab import Vocabulary
from torchvision import transforms
import time


def trainer(epochs=1):
    """parameters"""
    image_vector_size = 256
    embed_size = 100
    margin = 0.5
    batch_size = 10
    epochs = epochs
    vocab_size = 9956
    momentum = 0.9
    lr = 0.0001
    pad_len = 62
    num_workers = 2
    batch_size = 100


    """set model"""
    imageCNN = ImageCNN(image_vector_size=image_vector_size)
    matchCNN = MatchCNN(embed_size=embed_size,
                        image_vector_size=image_vector_size,
                        vocab_size=vocab_size,
                        pad_len=pad_len)

    if torch.cuda.is_available():
        print("cuda is available")
        imageCNN = imageCNN.cuda()
        matchCNN = matchCNN.cuda()

    """load models"""
    # model_path = "../models"

    #   imageCNN.load_state_dict(torch.load(os.path.join(model_path, 'imageCNN0.pkl')))
    #    matchCNN.load_state_dict(torch.load(os.path.join(model_path, 'matchCNN0.pkl')))
    """set optimizer"""
    # params = list(imageCNN.parameters()) + list(matchCNN.parameters())
    # params = list(imageCNN.linear.parameters()) + list(imageCNN.bn.parameters()) + list(matchCNN.parameters())
    params = list(imageCNN.parameters()) + list(matchCNN.parameters())
    optimizer = optim.SGD(params, momentum, lr)

    # Load vocabulary wrapper.
    with open("../data/coco/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

        # Image preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    """load train data"""
    # Build data loader
    data_loader = get_loader(root="../data/coco/resized2014",
                             json="../data/coco/annotations/captions_train2014.json",
                             vocab=vocab,
                             transform=transform,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers,
                             pad_len=pad_len)

    mean_losses = []
    start = time.time()
    target = Variable(torch.ones(batch_size, 1)).cuda()
    losses = []
    imageCNN.train()
    matchCNN.train()
    for epoch in range(epochs):
        losses = []
        for i, (images, captions, lengths) in enumerate(data_loader):
            """input data"""
            #         image = Variable(torch.randn(batch_size,3,224,224))
            #         sentences = Variable(torch.LongTensor(np.random.randint(low=0, high=999, size=(batch_size,pad_len))))
            if images.size(0) != batch_size:
                break

            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
            images = Variable(images, volatile=True)
            captions = Variable(captions)
            imageCNN.zero_grad()
            matchCNN.zero_grad()

            """extract imgae feature and embed sentence"""
            #         imageCNN = imageCNN.cuda()
            image_vectors = imageCNN(images)
            #         print(image_vectors)
            if torch.cuda.is_available():
                image_vectors_wrong = image_vectors[(torch.randperm(batch_size)).cuda()]
            else:
                image_vectors_wrong = image_vectors[torch.randperm(batch_size)]


                #         """get correct score"""
            scores = matchCNN(image_vectors, captions)
            scores_wrong = matchCNN(image_vectors_wrong, captions)
            #         print("scores",scores)

            #         break
            lossFunc = torch.nn.MarginRankingLoss(margin=0.5)
            #         loss = torch.max(margin + scores_wrong - scores, 0)
            loss = lossFunc(scores, scores_wrong, target)
            losses.append(loss)
            loss.backward()
            optimizer.step()
            #if i % 1000 == 0:
            #    print("i:%s,loss:%s"%(i,loss))
            #             print("time used:", time.time() - start)
            #if i == 1:
             #   print("time used:", time.time() - start)
              #  break
        mean_loss = torch.mean(torch.cat((losses)))
        mean_losses.append(mean_loss)
        print("epoch:", epoch)
        print("mean loss:", mean_loss)
        model_path = "../models"
        """save models"""
        torch.save(imageCNN.state_dict(),
                   os.path.join(model_path, 'imageCNN%d-%d-%f.pkl' % (start, epoch, mean_loss.cpu().data.numpy())))
        torch.save(matchCNN.state_dict(),
                   os.path.join(model_path, 'matchCNN%d-%d-%f.pkl' % (start, epoch, mean_loss.cpu().data.numpy())))

    print("time used:", time.time() - start)


def main(args):
    trainer(args.epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', type=str, default=None,
                        help='model file to load')
    parser.add_argument('--epochs', type=int, default=1,
                        help='epochs for train')

    args = parser.parse_args()
    main(args)