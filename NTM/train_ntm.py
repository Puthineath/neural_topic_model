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

@click.command()
@click.option('--epochs', default=20, prompt='The number of epochs',
              help='The number of epochs we train the model for.', show_default=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# prepare data

# Load data

# training model

ntm = NeuralTopicModel()

    # def cost_func(word, d_pos, d_neg):
        # omega = 0.5
        # return max(0,omega-ls(g,d_pos)+ls(g,d_neg))
    # if g is in d:
            # priority
    # else:

# training loop

    # if

    # optimizer SGD
    # criterion

    #for epoch in range(epochs):


# -----------------How to train a model?---------------

# Define the architecture (neural_topic_model.py)
# Forward propagate on the architecture using input data
# Calculate the loss
# Backpropagate to calculate the gradient for each weight
# Update the weights using a learning rate