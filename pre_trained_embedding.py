import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import re
import string
from gensim.models import CoherenceModel
from spacy.lang.en import English
import random
import pandas as pd
from gensim import corpora
import pickle

parser = English()
# nltk.download('stopwords')

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

def pre_processing(text):
    # en_stop = set(nltk.corpus.stopwords.words('english'))
    en_stop = nltk.corpus.stopwords.words('english')
    # add more stop words mostly appear in the texts
    en_stop.extend(['programme', 'accordance', 'article', 'state', 'member', 'this', 'annex', 'paragraph'])
    # remove numbers and non-word
    text = re.sub(r'\W*\d+\W*|\W+\w+\W', ' ', text)
    # # tokenize the text
    tokens = tokenize(text)
    # # get the words whose length are greater than 4 characters
    tokens = [token for token in tokens if len(token) > 4]
    # # remove stop words
    tokens = [token for token in tokens if token not in en_stop]
    # # get lemma
    tokens = [get_lemma(token) for token in tokens]

    return tokens

def get_data():
    text_data = []
    data = pd.read_csv("data/data1.csv")
    features = data['title'] + " " + data['article']
    for feature in features:
        tokens = pre_processing(feature)

        text_data.append(tokens)
    return text_data

def word2vec():
    name = "data/GoogleNews-vectors-negative300.bin.gz"
    # Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
    model = KeyedVectors.load_word2vec_format(name, binary=True)

    dog = model['authorize']
    print(dog.shape)
    print(f'10 dim of dog:\n {dog}')
    # cat = model['cat']
    # print(f'10 dim of cat:\n {cat[:10]}')
    # human = model['human']
    # print(f'10 dim of human:\n {human[:10]}')

    return model

def main():

    clean_words_doc = get_data()
    # dictionary is vocab (vocab refer to the distinct tokens)
    dictionary = corpora.Dictionary(clean_words)
    corpus = [dictionary.doc2bow(text) for text in clean_words_doc]
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')
    # print(corpus)
    # print(dictionary)
    model_word2vec = word2vec()

    out_of_vocab = []
    out_of_vocab_word = []

    for word in dictionary.token2id:
        try:
            word = model_word2vec[word]
            # print(word)
        except:
            out_of_vocab_word.append(word)
            word = 0
            out_of_vocab.append(word)
    print(len(out_of_vocab))
    print(out_of_vocab_word)
    #     if word in






if __name__ == '__main__':
    # main()
    word2vec()
