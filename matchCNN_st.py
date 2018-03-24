import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class MatchCNN_st(nn.Module):
    def __init__(self, embed_size, image_vector_size, vocab_size, pad_len, stride=3, conv1=200, conv2=300, conv3=300,
                 linear2=400):
        super(MatchCNN_st, self).__init__()
        self.stride = 3
        linear1_input = pad_len
        for i in range(3):
            linear1_input = (linear1_input - stride + 1) / 2
        linear1_input = int(linear1_input)
        linear1_input *= conv3

        self.embed = nn.Embedding(vocab_size, embed_size)

        self.conv1_sen = nn.Linear(embed_size * stride, conv1)
        self.conv2_sen = nn.Linear(conv1 * stride, conv2)
        self.conv3_sen = nn.Linear(conv2 * stride, conv3)
#         self.bn1_sen = nn.BatchNorm1d(linear1_input)

        self.muti_linear1_sen = nn.Linear(linear1_input + image_vector_size, linear2)
        # self.bn2_sen = nn.BatchNorm1d(linear2)
        self.linear2_sen = nn.Linear(linear2, 1)
        # self.bn3_sen = nn.BatchNorm1d(1)

        #self.init_weight()

    def init_weight(self):
        
        self.muti_linear1_sen.weight.data.normal_(0.0, 0.02)
        self.muti_linear1_sen.bias.data.fill_(0)

        self.linear2_sen.weight.data.normal_(0.0, 0.02)
        self.linear2_sen.bias.data.fill_(0)

    """
        image_vectors: batch_size * sentence_vector_size
        sentences : batch_size * sentence_size(now fixed as 30)
        note: Every image_vector and sentences pair should be matched
    """

    def forward(self, image_vectors, sentences):
        # For test only
        #         self.sentence_vectors = Variable(torch.randn((10, 30, 50)), requires_grad = True)
        #         image_vectors = Variable(torch.randn(10, 256))

        sentence_vectors = self.embed(sentences)
        features_sen = self.conv(sentence_vectors, self.conv1_sen)
        features_sen = self.conv(features_sen, self.conv2_sen)
        features_sen = self.conv(features_sen, self.conv3_sen)
        features_sen = self.mlp(features_sen, self.muti_linear1_sen, self.linear2_sen, image_vectors)

        return features_sen

    """
    features:  batch_size * sentence_size * channel_size
    return scores: batch_size * 1
    """

    def mlp(self, features, linear_function1, linear_function2, image_vectors=None):
        features = features.contiguous()
        features_num = self.num_flat_features(features)
        #   print("flat size:", features_num)
        features = features.view(-1, features_num)
        # features = self.bn1_sen(features)

        if image_vectors is not None:
            features = torch.cat([features, image_vectors], dim=1)
        features = F.leaky_relu(linear_function1(features))

        # features = self.bn2_sen(F.leaky_relu(linear_function1(features)))
        features = linear_function2(features)

        # features = self.bn3_sen(linear_function2(features))
        #   print("final shape:", features.data.numpy().shape)
        return features
    #     def muti_mlp(self, features, image_vectors, linear_function1, linear_function2):
    #         features = features.contiguous()
    #         features_num = self.num_flat_features(features)
    #         print("flat size:", features_num)

    #         features = features.view(-1, features_num)
    #         features = torch.cat([features,image_vectors], dim=1)
    #         features = F.relu(linear_function1(features))
    #         features = F.relu(linear_function2(features))
    #         print("final shape:",features.data.numpy().shape)
    #         return features

    """
    includ convlution, zero_gate and pooling
    """
    #     def muti_conv(self, features, image_vectors, muti_conv_function):
    #         features1 = self.scan_conv(features, image_vectors)
    #         features = F.relu(muti_conv_function(features1))
    #         features = self.zero_gate(features1, features)
    #         print("muti_convlution1 features shape:", features.size())
    #         features = self.sentence_pooling(features)
    #         return features;
    def conv(self, features, conv_function, image_vectors=None):
        features1 = self.scan_conv(features, image_vectors)
        features = F.leaky_relu(conv_function(features1))
        features = self.zero_gate(features1, features)
        # print("no zero gate")
        #   print("muti_convlution1 features shape:", features.size())
        features = self.sentence_pooling(features)
        return features

    """
    x: batch_size * feature1_size *... * featuren_size
    return: feature1_size * feature2_size * .... featuren_size
    """

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    """
    features: batch_size * sentence_size * channel_size
    return: batch_size * sentence_size/2 * channel_size
    """

    def sentence_pooling(self, features):
        return (F.max_pool1d(features.permute(0, 2, 1), 2)).permute(0, 2, 1)

    """
    features: batch_size * sentence_size * channel_size
    image_vectors: batch_size * image_size
    sentence_image_vectors: batch_size * (sentence_size - stride +1) * (stride*channel_size + image_size)
    """
    #     def scan_muticonv(self, features, image_vectors):
    #         batch_size = features.size(0)
    #         sentence_size = features.size(1)
    #         channel_size = features.size(2)
    #         image_size = image_vectors.size(1)
    #         print("muti_convlution input features shape:", features.size())
    # #         features_transpose = features.permute(0, 2, 1)

    #         sentence_image_vectors = Variable(torch.FloatTensor(batch_size, sentence_size - 3 + 1, 3*channel_size + image_size))
    #         print("sentence_image_vectors shape:", sentence_image_vectors.size())
    #         for i in range(3):
    #             sentence_image_vectors[:,:,i * channel_size:(i+1)*channel_size] = features[:,i:sentence_size - 3 + 1 + i,:]
    #         sentence_image_vectors[:,:,3*channel_size:] = image_vectors.unsqueeze(1).repeat(1, sentence_size - 3 + 1,1)

    # #       features = self.muti_conv1(sentence_image_vectors)
    #         return sentence_image_vectors


    """
    features: batch_size * sentence_size * channel_size
    sentence_image_vectors: batch_size * (sentence_size - stride +1) * (stride * channel_size)
    """

    #     def scan_conv(self, features):
    #         batch_size = features.size(0)
    #         sentence_size = features.size(1)
    #         channel_size = features.size(2)
    #         image_size = image_vectors.size(1)
    #         print("muti_convlution input features shape:", features.size())
    #         #         features_transpose = features.permute(0, 2, 1)

    #         sentence_vectors = Variable(torch.FloatTensor(batch_size, sentence_size - 3 + 1, 3*channel_size))

    #         print("sentence_vectors shape:", sentence_vectors.size())
    #         for i in range(3):
    #             sentence_vectors[:,:,i * channel_size:(i+1)*channel_size] = features[:,i:sentence_size - 3 + 1 + i,:]
    #         return sentence_vectors
    def scan_conv(self, features, image_vectors=None):
        stride = self.stride
        batch_size = features.size(0)
        sentence_size = features.size(1)
        channel_size = features.size(2)
        #  print("muti_convlution input features shape:", features.size())
        #         features_transpose = features.permute(0, 2, 1)
        if image_vectors is None:
            sentence_vectors = Variable(
                torch.FloatTensor(batch_size, sentence_size - stride + 1, stride * channel_size))
        else:
            image_size = image_vectors.size(1)
            sentence_vectors = Variable(
                torch.FloatTensor(batch_size, sentence_size - stride + 1, stride * channel_size + image_size))
        if torch.cuda.is_available():
            sentence_vectors = sentence_vectors.cuda()
            #  print("sentence_vectors shape:", sentence_vectors.size())
        for i in range(stride):
            sentence_vectors[:, :, i * channel_size:(i + 1) * channel_size] = features[:,
                                                                              i:sentence_size - stride + 1 + i, :]
        if image_vectors is not None:
            sentence_vectors[:, :, stride * channel_size:] = image_vectors.unsqueeze(1).repeat(1,
                                                                                               sentence_size - stride + 1,
                                                                                               1)
        return sentence_vectors

    """
    if vector in feature1 is zero vectors, vector in feature should also be zero
    """

    def zero_gate(self, feature1, feature2):
        zero_vectors = feature1.sum(dim=2, keepdim=True)
        zero_vectors = (zero_vectors != 0).float()
        # zero_vectors[zero_vectors != 0] = 1
        return torch.mul(feature2, zero_vectors)

