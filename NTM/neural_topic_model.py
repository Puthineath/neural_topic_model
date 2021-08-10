import torch
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


def n_gram_topic_layer(word,weight_k_topic):
    """
    :param word: word (1 x 300)
    :param weight_k_topic: word x topic = 300 x k (topic = k)
    :return: topic_layer 1 x k
    """
    result = torch.mm(word, weight_k_topic)
    return torch.sigmoid(result)



def n_gram_topic_layer2(le,weight_k_topic):
    le2 = torch.mm(le,torch.transpose(weight_k_topic,0,1))

    return torch.sigmoid(le2)

# def topic_document_layer(document):
#     weight_1 =
#     return torch.softmax(weight_1)

if __name__ == '__main__':

    # need to load the embedding data

    le = n_gram_topic_layer(torch.randn(2, 3), torch.randn(3, 3))
    le2=n_gram_topic_layer2(le,torch.randn(3, 3))
    print(f"le:\n {le}")
    print(f"le':\n {le2}")

    # need to optimize le2 to get close to le, can tune w2










