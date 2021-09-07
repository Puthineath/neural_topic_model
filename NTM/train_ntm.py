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

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_ntm():
    ntm_model = NeuralTopicModel()
    w1 = ntm_model.w1
    w2 = ntm_model.w2
#------------------------------------------prepare data---------------------------------------
    d_pos_list_len = []
    single_data_pos_list = []
    single_data_neg_list = []

    # get each element of positive doc
    for i in data.positive:
        for g, d_pos_list in i.items():
            d_pos_list_len.append(len(d_pos_list))
            for d_pos in d_pos_list:
                single_data_pos_list.append({g:d_pos})

    # get random d_neg of negative docs based on the length of each length of positive docs
    for index,j in enumerate(data.negative):
        for k in range(d_pos_list_len[index]):
            for g1,d_neg_list in j.items():
                if d_neg_list == []:
                    single_data_neg_list.append({g1: ''})
                else:
                    single_data_neg_list.append({g1: random.choice(d_neg_list)})

    # put all 3 elements into one list and then put all the lists into final_data_list
    word_dpos_list = []
    final_data_list = []
        # put only positive value in the final_data_list first
    for single_data_pos in single_data_pos_list:
        for word, d_pos in single_data_pos.items():
            word_dpos_list.append(word)
            word_dpos_list.append(d_pos)
            final_data_list.append(word_dpos_list.copy())
            word_dpos_list.clear()
        # add negative document into the final_data_list
    for index,single_data_neg in enumerate(single_data_neg_list):
        for word,d_neg in single_data_neg.items():
            final_data_list[index].append(d_neg)


    # ---------------------------------------training----------------------------------------------------
        # Stochastic Gradient Decent with values of learning rates and regularization factor mentioned in the paper
    optimizer = torch.optim.SGD(params= ntm_model.parameters(), lr=0.01,weight_decay= 0.001)
        # Mean Square Root metric to compute the loss
    criterion = nn.MSELoss()

    epochs = 5
    losses_list = []

    for i in range(0, epochs):
        losses = []

        # for elt in final_data_list:
        for elt in final_data_list[1:5]: # test 5 data in the list
            word = elt[0]
            d_pos = elt[1]
            d_neg = elt[2]

            # get only the number from 'id_0' => '0'
            d_pos = int(' '.join(re.findall("\d+", d_pos)))
            # change dimension from 5 to 1 x 5
            d_pos = torch.unsqueeze(w1[d_pos], 0) # get the row of matrix w1 correspond to Id number

            if d_neg != '':
                d_neg = int(' '.join(re.findall("\d+", d_neg)))
                d_neg = torch.unsqueeze(w1[d_neg],0)

    # --------------------------------------calculate cost function and back propagate -------------------------------------
    #         print(ntm_model.cost_func(word,d_pos,d_neg))

            optimizer.zero_grad()

            if ntm_model.cost_func(word,d_pos,d_neg) > 0:
                predictions = ntm_model(word, d_pos)
                # I put the target value to 1 because I think the maximum value of probability is 1 and then I put it at the same size of the prediction
                target = torch.ones(predictions.size())
                loss = criterion(predictions, target)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            else:
                # if cost_function == 0, add 0 as value of loss
                losses.append(0)
        losses_list.append(mean(losses))
    # save model
    torch.save({"state_dict": ntm_model.state_dict(), "losses": losses_list},
                   f'../models/ntm_model_train_{epochs}epochs.pth')
if __name__ == '__main__':
    print(train_ntm())



