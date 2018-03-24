import torch
from torch.autograd import Variable
import torch.optim as optim
from model import ImageCNN, MatchCNN
import argparse
import os
from data_loader import get_loader
from torchvision import transforms
import time
from matchCNN_st import MatchCNN_st
import pickle
import numpy as np
from build_vocab import Vocabulary


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def trainer(epochs=1):
    """parameters"""
    image_vector_size = 256
    embed_size = 100
    margin = 0.1
    batch_size = 100
    epochs = 50
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
        imageCNN.cuda()
        matchCNN.cuda()

    """load models"""
    model_path = "../models"

#     imageCNN.load_state_dict(torch.load(os.path.join(model_path, 'imageCNN_Nobn&drop_st90-0.005311.pkl')))
#     matchCNN.load_state_dict(torch.load(os.path.join(model_path, 'matchCNN_Nobn&drop_st90-0.005311.pkl')))

    # imageCNN.eval()
    # matchCNN.eval()
    """set optimizer"""
    params = list(imageCNN.parameters()) + list(matchCNN.parameters())
    # params = list(imageCNN.linear.parameters()) + list(imageCNN.bn.parameters()) + list(matchCNN.parameters())
    # params = list(imageCNN.parameters()) + list(matchCNN.parameters())
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

    start = time.time()
    for epoch in range(90):
        losses = []
        corrects = []
        for i, (images, images_wrong, captions, lengths) in enumerate(data_loader):
            """input data"""
    #         image = Variable(torch.randn(batch_size,3,224,224))
    #         sentences = Variable(torch.LongTensor(np.random.randint(low=0, high=999, size=(batch_size,pad_len))))
    #         if images.size(0) != batch_size:
                # break

            images = to_var(images, volatile=True)
            images_wrong = to_var(images_wrong, volatile=True)
            captions = to_var(captions)

            imageCNN.zero_grad()
            matchCNN.zero_grad()

            """extract imgae feature and embed sentence"""
            image_vectors = imageCNN(images)
            image_vectors_wrong = imageCNN(images_wrong)

            """get correct score"""
            scores = matchCNN(image_vectors, captions)
            scores_wrong = matchCNN(image_vectors_wrong, captions)
            batch_size = images.data.shape[0]
            target = to_var(torch.ones(batch_size, 1)).cuda()

            """get loss"""
            mrl = torch.nn.MarginRankingLoss(margin)
            loss = mrl(scores, scores_wrong, target)

            # loss = torch.sum(scores_wrong - scores + margin)

            loss.backward()

            """update"""     
            optimizer.step()
            losses.append(loss)
            d = (scores > scores_wrong).cpu().data
            e = torch.sum(d)
            correct = e / 100 * 100
            corrects.append(correct)
            if i % 100 == 0:
                print("-"*30)
#                 print("score:", scores[0])
#                 print("score_wrong:", scores_wrong[0])

                print("epoch:%s, i:%s,loss:%s" % (epoch, i*batch_size, loss))
                print("correct rate:%s" % np.mean(corrects))

    #         if i == 100:
    #             break
    #             print("scores",scores[0])
        mean_loss = torch.mean(torch.cat(losses))
        print("epoch:%s,mean loss:%s" % (epoch, mean_loss))
        
        if (epoch + 1)%5 is 0:
            model_path = "../models"
            """save models"""
            torch.save(imageCNN.state_dict(), os.path.join(model_path, 'imageCNN_mar0.5_st%s-%f.pkl' % (epoch, mean_loss.cpu().data.numpy())))
            torch.save(matchCNN.state_dict(), os.path.join(model_path, 'matchCNN_mar0.5_st%s-%f.pkl' % (epoch, mean_loss.cpu().data.numpy())))


        print("time used:", time.time() - start)

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