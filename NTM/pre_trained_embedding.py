"""
    - Pre-process data
    - Get the unique tokens from datasets
    - Match the word in dataset to the embedding from word2vec
    - Get the dictionary {word:doc} of words whether it appear in each document

"""

from gensim.models import KeyedVectors
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import re
from spacy.lang.en import English
import pandas as pd
from gensim import corpora
import pickle
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import numpy as np
import json
from json import JSONEncoder
import csv
parser = English()
# nltk.download('stopwords')

# pre_process data
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

# get data from the file
def get_data():
    text_data = []
    data = pd.read_csv("../data/data1.csv")
    features = data['title'] + " " + data['article']
    for feature in features:
        tokens = pre_processing(feature)
        text_data.append(tokens)
    return text_data

clean_words_doc = get_data()

# open the word2vec pretrained embedding file
def word2vec():
    name = "C:/Users/salbo/puthineath/eurovoc_conversion/eurovoc_conversion/data/GoogleNews-vectors-negative300.bin.gz"
    # Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
    model = KeyedVectors.load_word2vec_format(name, binary=True)
    # dog = model['dog']
    # print(dog.shape)
    # print(f'10 dim of dog:\n {dog}')
    return model

# assign the id to the document
def doc_id():
    doc_dict_list=[]
    id = 0
    for doc in clean_words_doc:
        doc_dict = {f'id_{id}':doc}
        id +=1
        doc_dict_list.append(doc_dict)
    return doc_dict_list

# check if the a word in the document or not
# doc positive
"""
    sameple doc : (word, doc)
"""
def sample_doc_positive(word):
    word_in_doc_list = []
    word_not_in_doc_list = []
    for i in doc_id():
        for key, value in i.items():
            if word in value:
                word_in_doc_dict = {word:key}
                word_in_doc_list.append(word_in_doc_dict)
            else:
                word_not_in_doc_dict = {word: key}
                word_not_in_doc_list.append(word_not_in_doc_dict)
    return word_in_doc_list
    # return word_in_doc_list,word_not_in_doc_list

# doc negative
def sample_doc_negative(word):
    word_in_doc_list = []
    word_not_in_doc_list = []
    for i in doc_id():
        for key, value in i.items():
            if word in value:
                word_in_doc_dict = {word:key}
                word_in_doc_list.append(word_in_doc_dict)
            else:
                word_not_in_doc_dict = {word: key}
                word_not_in_doc_list.append(word_not_in_doc_dict)
    return word_not_in_doc_list

# get dictionary. dictionary is vocab (vocab refer to the distinct tokens in the dataset)
def get_dictionary():
    dictionary = corpora.Dictionary(clean_words_doc)
    corpus = [dictionary.doc2bow(text) for text in clean_words_doc]
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')
    return dictionary

def list_of_sample_id_positive():
    sample_doc_list_positive = []
    for word in get_dictionary().token2id:
        sample_doc_list_positive.append(sample_doc_positive(word))
    return sample_doc_list_positive

def list_of_sample_id_negative():
    sample_doc_list_negative = []
    for word in get_dictionary().token2id:
        sample_doc_list_negative.append(sample_doc_negative(word))
    return sample_doc_list_negative

# merge values to the same key
def merge_list_of_dictionaries(dict_list):
    new_dict = {}
    for d in dict_list:
        for d_key in d:
            if d_key not in new_dict:
                new_dict[d_key] = []
            new_dict[d_key].append(d[d_key])
    return new_dict

def merge_value(list_of_sample_id):
    merge_value_list=[]
    for dict in list_of_sample_id:
        merge_value_list.append(merge_list_of_dictionaries(dict))
    return  merge_value_list

# get the word and its embeddings
def get_word_embedding():
    model=word2vec()
    list_of_embedding = []
    for word in get_dictionary().token2id:
        try:
            list_of_embedding.append({word:model[word]})
        except:
            list_of_embedding.append({word: 0})
    return list_of_embedding

def save_txt():
    word_embedding = get_word_embedding()
    with open('C:/Users/salbo/puthineath/eurovoc_conversion/eurovoc_conversion/data/data_embeddings.txt', 'w') as f:
        for dict in word_embedding:
            for word, embed in dict.items():
                # if type(embed) == np.array:
                #     embed = embed.tolist()
                #     embed = ''.join(str(e) for e in embed)
                #     f.write(f"{word} {embed}\n")
                # else:
                f.write(f"{word} {embed}\n")

def main():
    print(f'Words appear in documents (d_pos):\n{merge_value(list_of_sample_id_positive())}\n')
    print(f'Words do not appear in documents (d_neg):\n{merge_value(list_of_sample_id_negative())}')
    save_txt() # already saved in the data folder
    return

if __name__ == '__main__':
    main()






