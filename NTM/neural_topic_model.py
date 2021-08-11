import pandas as pd
import torch
from pre_trained_embedding import word2vec, doc_id
import csv
import torch.nn.functional as f
"""
    create neural topic model based on Cao et al.
"""
# import numpy as np
#
# class NeuralTopicModel():
#     def __init__(self,g,d):
#         """
#         :param g: word
#         :param d: document
#         """
#         self.g = g
#         self.d = d
data_path = 'C:/Users/salbo/puthineath/eurovoc_conversion/eurovoc_conversion/data/clean_docs.csv'

# get le
def n_gram_topic_layer(word,weight_k_topic):
    """
    :param word: word (1 x 300)
    :param weight_k_topic: word x topic = 300 x k (topic = k)
    :return: topic_layer 1 x k
    """
    result = torch.mm(word, weight_k_topic)
    return torch.sigmoid(result)

# get le'
def n_gram_topic_layer2(le,weight_k_topic):
    le2 = torch.mm(le,torch.transpose(weight_k_topic,0,1))
    return torch.sigmoid(le2)

# load the word2vec
def load_word(word):
    model_embeddings = word2vec()
    try:
        embedding = model_embeddings[word]
    except:
        embedding = 0
    return embedding


def main():
    documents = doc_id()
    dog_embed = load_word('dog')
    # for document in documents:
    #     for id,tokens in document.items():
    #         for word in tokens:
    #             embedding = load_word(word[0])

    w2 = torch.randn(300, 5)
    le = n_gram_topic_layer(torch.from_numpy(dog_embed), w2)
    le2=n_gram_topic_layer2(le,w2)
    print(f"le:\n {le}")
    print(f"le':\n {le2}")
    return

# def tensor():
#     dog_embed = load_word('dog')
#     print(type(torch.from_numpy(dog_embed)))

if __name__ == '__main__':
    main()



    # need to load the embedding data


    # need to optimize le2 to get close to le, can tune w2










