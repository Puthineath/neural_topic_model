import click
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import random as rd
from functools import partial
from statistics import mean
from sklearn.model_selection import train_test_split
from copy import copy
from neural_topic_model import NeuralTopicModel
from pre_trained_embedding import *


# @click.command()
# @click.option('--epochs', default=10, prompt='The number of epochs',
#               help='The number of epochs we train the model for.', show_default=True)
# @click.option('--topics', default=5, prompt='The number of topics',
#               help='The number of pre-defined topics we want', show_default=True)
# # device = "cuda" if torch.cuda.is_available() else "cpu"

# prepare data
#
# Load data
#
# need to split data for training and testing

def cos(g, d_pos=torch.randn(1,5), d_neg=torch.randn(1,5), model_ntm=NeuralTopicModel()):
    omega = 0.5
    a = model_ntm.forward(g, d_pos)
    b = model_ntm.forward(g, d_neg)
    result = max(0, (omega - a + b))
    return result


# training model

def negative():
    negative_doc = []

    for i in list_of_sample_id_positive():
        for doc in i:
            negative_doc.append(doc)
    return negative_doc


def positive():
    positive_doc = []

    for i in list_of_sample_id_positive():
        for doc in i:
            positive_doc.append(doc)
    return positive_doc

# list of dict(word,doc) of words appear document
positive = positive()
# list of dict(word,doc) of words do not appear document
negative = negative()


# def train_ntm(topics, epochs):
def train_ntm():
    model_ntm = NeuralTopicModel()

    # optimizer = torch.optim.SGD(model_ntm.parameters(), lr=0.01, weight_decay=0.001)
    optimizer = torch.optim.SGD(model_ntm.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # to be able to add other losses, which are tensors, we initialize the loss as a 0 tensor
    loss = torch.tensor(0).float()

    # losses = []

    # load the data
    for pair in positive:
        # if cos(word for word,value in pair.items()) > 0:

        for word, value in pair.items():
            for pair_neg in negative:
                for word_neg, value_neg in pair_neg.items():

                    if word == word_neg:
                        d_neg = torch.randn(1, 5)
                    else:
                        # if word in positive list not appear in negative list, so put it as 0 (because minus 0)
                        d_neg = torch.zeros(1,5)

            d_pos = torch.randn(1, 5)
            if cos(word, d_pos=d_pos, d_neg=d_neg) > 0:
                optimizer.zero_grad()
                # # to be able to add other losses, which are tensors, we initialize the loss as a 0 tensor
                # loss = torch.tensor(0).to(device).float()
                output = model_ntm(word, d_pos)
                target = torch.ones(output.size())
                print(output)
                print(f'loss before {loss}')
                print(target)
                loss += criterion(output, target)
                print(f'loss after {loss}')
    #
    # loss.backward()
    #
    # optimizer.step()
    # print(loss)
    # return loss


if __name__ == '__main__':
    train_ntm()
    # output = cos('dog')
    # print(output)
