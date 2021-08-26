import torch, torch.nn as nn
import re
from pre_trained_embedding import *
from torch.utils.data import Dataset, DataLoader

# list of dict(word,doc) of words appear document
positive = merge_value(list_of_sample_id_positive())
# list of dict(word,doc) of words do not appear document
negative = merge_value(list_of_sample_id_negative())


class DataTask(Dataset):
    def __init__(self,positive,negative):
        super(DataTask).__init__()
        self.positive = positive
        self.negative = negative

    def __len__(self):
        return len(self.positive)
    def get(self,index):
        dict_pos_list = positive[index]
        dict_neg_list = negative[index]
        # for word,ids_pos in dict_pos_list.items():
        return dict_pos_list, dict_neg_list

    # def __getitem__(self, index):
    #     dict_pos_list = positive[index]
    #     dict_neg_list = negative[index]
    #     # for word,ids_pos in dict_pos_list.items():
    #
    #     return dict_pos_list,dict_neg_list


data = DataTask(positive,negative)

if __name__ == '__main__':
    print(len(data.positive))
    print(data.get(1))



# Pre-train
# create w1 initialization: 4 x 5
# create w2 initialization: 3 x k

# Create T and not_T
# Loop to Update W1 and W2 :
# sample one document in not_T
# Calculate ls for pos
# Calculate ls for neg
# cost_function = result








