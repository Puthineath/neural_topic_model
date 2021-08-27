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
from data import DataTask
import random

# list of dict(word,doc) of words appear document
positive = merge_value(list_of_sample_id_positive())
# list of dict(word,doc) of words do not appear document
negative = merge_value(list_of_sample_id_negative())

data = DataTask(positive,negative)

def get_word(dict):
    for word, ids in dict.items():
        return word

def get_id_pos(dict):
    for word,ids_pos in dict.items():
        for id_pos in ids_pos:
            return id_pos

def get_id_neg(dict):
    if dict == {}:
        return 'id_0' #***** need to check this
    else:
        for word, ids_neg in dict.items():
            return random.choice(ids_neg)

def get_data(pos_neg_data):
    positive_dict, negative_dict = pos_neg_data
    return get_word(positive_dict), get_id_pos(positive_dict), get_id_neg(negative_dict)

# get cost function
def get_cost_func():
    model = NeuralTopicModel()  # use random size of document
    w1 = model.w1
    w2 = model.w2
    for i in range(len(data.positive)):
        word, d_pos, d_neg = get_data(data.get(i))
        # for word, d_pos, d_neg in get_data(data.get(i)):
        id_pos = int(' '.join(re.findall("\d+", d_pos))) #get only the number from 'id_0' => '0'
        id_neg = int(' '.join(re.findall("\d+", d_neg)))
        # change dimension from 5 to 1 x 5
        d_pos = torch.unsqueeze(w1[id_pos],0)
        d_neg = torch.unsqueeze(w1[id_neg],0)
        print(model.cost_func(word,d_pos,d_neg))

if __name__ == '__main__':
    print(get_cost_func())
    # the problem here is getting only the first element of the document is the positive list
    # ***** need to change the model to loop everything

