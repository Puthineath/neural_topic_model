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

data_path = 'C:/Users/salbo/puthineath/eurovoc_conversion/eurovoc_conversion/data/clean_docs.csv'

class NeuralTopicModel(nn.Module):
    """
    torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
    """
    def __init__(self, topic=5):
        super().__init__()
        self.w2 = torch.randn(300, topic) #  300 x 5 (topic k = 5)
        self.w1 = torch.randn(4, topic)  #  4 x 5 # number of documents is 4
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

    def ls(self,word,doc_id):
        word = self.load_word(word)
        # change tensor from size (300) to (1 x 300)
        input = torch.unsqueeze(self.array2tensor(word), 0)
        # get lt(g) by multiplying the matrix of input word and w2
        lt = torch.sigmoid(torch.mm(input,self.w2))
        # # get the ld(d)
        ld = F.softmax(doc_id, dim=-1)
        # # get the score layer (ls = lt x ld.T)
        ls = torch.dot(torch.reshape(lt, (-1,)), torch.reshape(torch.transpose(ld, 0, 1), (-1,)))
        return ls

    def cost_func(self,g, d_pos, d_neg):
        omega = 0.5
        a = self.ls(g, d_pos)
        b = self.ls(g, d_neg)
        result = max(0, (omega - a + b))
        return result

    def forward(self,word,d_pos,d_neg):
        cos = self.cost_func(word,d_pos,d_neg)
        return cos


def main():
    model = NeuralTopicModel() # use random size of document
    # test
    print(model.cost_func('dog',torch.randn(1,5),torch.randn(1,5)))
    print(model.ls('dog',torch.randn(1,5)))

if __name__ == '__main__':
    main()














