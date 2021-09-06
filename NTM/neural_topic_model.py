import pandas as pd
import torch
from pre_trained_embedding import doc_id
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

    def __init__(self, topic=5):
        super().__init__()
        self.w2 = torch.randn(300, topic) #  300 x 5 (topic k = 5)
        self.w1 = torch.randn(4, topic)  #  4 x 5 # number of documents is 4
        self.linear = nn.Linear(1,1)

    # load the word2vec
    def load_word(self,word):
        name = "../data/GoogleNews-vectors-negative300.bin.gz"
        # Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
        model_embeddings = KeyedVectors.load_word2vec_format(name, binary=True)
        try:
            embedding = model_embeddings[word]
        except:
            embedding = 0
        return embedding

    def array2tensor(self,array):
        return torch.from_numpy(array)

# find score layer
    # put these argument in the __init__() in order to update the weight
    def ls(self,word,doc_id):
        le = self.load_word(word)
        # change tensor from size (300) to (1 x 300)
        le = torch.unsqueeze(self.array2tensor(le), 0)
        # get lt(g) by multiplying the matrix of input word and w2
        lt = torch.sigmoid(torch.mm(le,self.w2))
        # get the ld(d)
        ld = F.softmax(doc_id, dim=1)
        # get the score layer (ls = lt x ld.T)
        ls = torch.dot(torch.reshape(lt, (-1,)), torch.reshape(torch.transpose(ld, 0, 1), (-1,)))
        return ls

# calculate cost_function
    def cost_func(self,g, d_pos, d_neg):
        # g = word
        # omega = 0.5
        if d_neg == '':
            result = max(0, (0.5 - self.ls(g, d_pos)))
        else:
            result = max(0, (0.5 - self.ls(g, d_pos) + self.ls(g, d_neg)))
        return result

# find the topic representation
    def forward(self,word,d_pos):

        return self.linear(torch.unsqueeze(self.ls(word,d_pos), 0))

def main():
    ntm_model = NeuralTopicModel()
    optimizer = torch.optim.SGD(params= ntm_model.parameters(), lr=0.01)
    # momentum
    criterion = nn.MSELoss()

    word = 'dog'
    d_pos = torch.randn(1,5)

    epochs = 3
    losses_list = []
    losses = []

    for i in range(0, epochs):
        optimizer.zero_grad()
        predictions = ntm_model(word,d_pos)
        # I put the target value to 1 because the maximum value of probab. is 1 and then I put it at the same size of the prediction
        target = torch.ones(predictions.size())
        loss = criterion(predictions, target)

        loss.backward()
        optimizer.step()

        print("-" * 10)
        print(f'epoch {i}:')
        print(f'loss:{loss}')
        # print("all = {}".format(list(ntm_model.parameters())))
        print("learned weight = {}".format(list(ntm_model.parameters())[0].data[0, 0]))
        print("learned bias = {}".format(list(ntm_model.parameters())[1].data[0]))

        losses.append(loss.item())
    torch.save({"state_dict": ntm_model.state_dict(), "losses": losses}, f'../models/ntm_model1.pth')

        # put loss into the list
        # save model

if __name__ == '__main__':
    main()














