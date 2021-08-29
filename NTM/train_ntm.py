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


# get cost function
def get_cost_func():
    model = NeuralTopicModel()  # use random size of document
    w1 = model.w1
    w2 = model.w2
    list_data = []
    single_data_pos_list = []
    single_data_neg_list = []
    for i in data.positive:
        for g, d_pos_list in i.items():
            for d_pos in d_pos_list:
                single_data_pos_list.append({g:d_pos})
    for j in data.negative:
        for g1,d_neg_list in j.items():
            # if d_neg_list == []:
                # single_data_neg_list.append({g1: None})
            # else:
            for d_neg in d_neg_list:

                # single_data_neg_list.append({g1: int(str(random.choice([d_neg for d_neg in d_neg_list if d_neg != None])))})
                single_data_neg_list.append({g1: random.choice(d_neg_list)})


    # zip dictionary and created to one list and then calculate cost_function
    #             - store value into a list


            # for d_pos in d_pos_list:
            #     for j in data.negative:
            #         for d_neg_list in j.values():
            #             try:
            #                 single_data_list.append({g:[d_pos,random.choice(d_neg_list)]})
            #             except:
            #                 single_data_list.append({g:[d_pos,None]})
            #                 # list_data.append(single_data_list)
            #             print(single_data_list)



                        # id_pos = int(' '.join(re.findall("\d+", single_data_list[]))) #get only the number from 'id_0' => '0'
                        # id_neg = int(' '.join(re.findall("\d+", d_neg)))
                        # # change dimension from 5 to 1 x 5
                        # d_pos_new = torch.unsqueeze(w1[id_pos],0)
                        # d_neg_new = torch.unsqueeze(w1[id_neg],0)



                print(list_data)




        # word, d_pos, d_neg = get_data(data.get(i))
        # for word, d_pos, d_neg in get_data(data.get(i)):


        # id_pos = int(' '.join(re.findall("\d+", d_pos))) #get only the number from 'id_0' => '0'
        # id_neg = int(' '.join(re.findall("\d+", d_neg)))
        # # change dimension from 5 to 1 x 5
        # d_pos = torch.unsqueeze(w1[id_pos],0)
        # d_neg = torch.unsqueeze(w1[id_neg],0)
        # print(model.cost_func(word,d_pos,d_neg))

if __name__ == '__main__':
    print(get_cost_func())
    # the problem here is getting only the first element of the document is the positive list
    # ***** need to change the model to loop everything

