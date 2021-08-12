import pandas as pd
import torch
from pre_trained_embedding import word2vec, doc_id
from gensim.models import KeyedVectors
import torch, torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv

"""
    create neural topic model based on Cao et al.
"""
# import numpy as np

# if g is in d:

# else:
data_path = 'C:/Users/salbo/puthineath/eurovoc_conversion/eurovoc_conversion/data/clean_docs.csv'

class NeuralTopicModel(nn.Module):
    """
    torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
    """
    # def __init__(self,word,doc):
    def __init__(self, word):
        super().__init__()
        self.word = word
        # self.doc = doc
        self.linear1 = nn.Linear(300, 5) # hidden layer 300 x 5 (topic k = 5)
        self.linear2 = nn.Linear(1, 5)  # hidden layer 1 x 5

    # load the word2vec
    def load_word(self,word):
        name = "C:/Users/salbo/puthineath/eurovoc_conversion/eurovoc_conversion/data/GoogleNews-vectors-negative300.bin.gz"
        # Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
        model_embeddings = KeyedVectors.load_word2vec_format(name, binary=True)
        try:
            embedding = model_embeddings[word]
        except:
            embedding = 0
        return embedding

    def array2tensor(self,array):
        return torch.from_numpy(array)

    def forward(self):
        # input the word and reshape from size (300) to (1 x 300)
        input = torch.reshape(self.array2tensor(self.load_word(self.word)), [1, 300])
        # get lt(g)
        lt = F.sigmoid(self.linear1(input))
        # get the ld(d)
        return lt

def main():
    model = NeuralTopicModel('dog')
    print(model.forward())

if __name__ == '__main__':
        main()

# #---------------------------manual-----------------------------------
# # get lt
#     def n_gram_topic_layer(word,weight_k_topic):
#         """
#         :param word: word (1 x 300)
#         :param weight_k_topic: word x topic = 300 x k (topic = k)
#         :return: topic_layer 1 x k
#         """
#         result = torch.mm(word, weight_k_topic)
#         return torch.sigmoid(result)
#
#     # get le'
#     def le2(word,weight_k_topic):
#         le2 = torch.mm(word,torch.transpose(weight_k_topic,0,1))
#         return torch.sigmoid(le2)
#
# # get ld(d)
#     def topic_doc_layer():
#         return
#
#     def score_layer():
#         return
#
# # def main_():
# #     documents = doc_id()
# #     dog_embed = load_word('dog')
# #     # for document in documents:
# #     #     for id,tokens in document.items():
# #     #         for word in tokens:
# #     #             embedding = load_word(word[0])
# #     word = torch.reshape(torch.from_numpy(dog_embed),[1,300])
# #     w2 = torch.randn(300, 5)
# #     lt = n_gram_topic_layer(word, w2)
# #     le_=le2(lt,w2)
# #
# #     print(f"lt:\n {lt}")
# #     print(f"lt:\n {lt.shape}")
# #
# #     print(f"le':\n {le_}")
# #     print(f"le':\n {le_.shape}")
# #     return












