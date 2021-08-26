import re

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


# cost function


# list of dict(word,doc) of words appear document
positive = merge_value(list_of_sample_id_positive())
# list of dict(word,doc) of words do not appear document
negative = merge_value(list_of_sample_id_negative())

# load the new data


# training model
# def train_ntm(topics, epochs):

def main():


    return
def train_ntm():
    model_ntm = NeuralTopicModel()

    optimizer = torch.optim.SGD(model_ntm.parameters(), lr=0.01, weight_decay=0.001)

    criterion = nn.MSELoss()

    # to be able to add other losses, which are tensors, we initialize the loss as a 0 tensor
    loss = torch.tensor(0).float()

    for pair in positive:
        for word,list_of_docs in pair.items():
            for i in range(len(list_of_docs)):
                print(i)

                output = model_ntm

    return



if __name__ == '__main__':
    # train_ntm()
    # output = cos('dog')
    # print(output)
    # print(test())

    print(get_data())