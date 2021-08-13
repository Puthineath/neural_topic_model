import torch, torch.nn as nn
import random

PATH = 'C:/Users/salbo/puthineath/eurovoc_conversion/eurovoc_conversion/data/clean_docs.txt'

def load_data(PATH):
    with open(PATH, "r", encoding="utf-8") as f:
        return [line.strip().split('\t') for line in f]


def encode_word():
    return

def decode_word():
    return

if __name__ == '__main__':
    print(load_data(PATH))