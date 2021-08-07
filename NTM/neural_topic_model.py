
"""
    create neural topic model based on Cao et al.
"""
# import numpy as np
#
class NeuralTopicModel():
    def __init__(self,g,d):
        """
        :param g: word
        :param d: document
        """
        self.g = g
        self.d = d
    def sigmoid(self, x):
        #applying the sigmoid function
        return 1 / (1 + np.exp(-x))


