"""
    - Get the unique tokens from datasets
    - Match the word in dataset to the embedding from word2vec
    - Get the dictionary {word:doc} of words whether it appear in each document
"""
from gensim.models import KeyedVectors
import pandas as pd
from gensim import corpora
import pickle
# nltk.download('wordnet')
import csv
from pre_processing import pre_processing
import pandas


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
    name = "C:/Users/salbo/puthineath/eurovoc_conversion/eurovoc_conversion/data/GoogleNews-vectors-negative300.bin.gz" # need to download this file
    # Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
    model = KeyedVectors.load_word2vec_format(name, binary=True)
    # dog = model['dog']
    # print(dog.shape)
    # print(f' 300 dim of dog:\n {dog}')
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


def save_clean_doc_id_csv(doc_dict_list):
    # open the file in the write mode
    with open('C:/Users/salbo/puthineath/eurovoc_conversion/eurovoc_conversion/data/clean_docs.csv', 'w') as f:
        # create the csv writer
        writer = csv.writer(f)
        for dictionary in doc_dict_list:
            for key, value in dictionary.items():
                writer.writerow([key, value])
        return


def save_clean_doc_id_txt(doc_dict_list):
    with open('C:/Users/salbo/puthineath/eurovoc_conversion/eurovoc_conversion/data/clean_docs.txt', 'w') as f:
        # create the csv writer
        for dictionary in doc_dict_list:
            for id, tokens_list in dictionary.items():
                f.write(f"{id} {tokens_list}\n")
        return


# check if the a word in the document or not
# doc positive
"""
    sample doc : {word:doc}
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
    with open('data/data_embeddings.txt', 'w') as f:
        for dict in word_embedding:
            for word, embed in dict.items():
                f.write(f"{word} {embed}\n")


def save_embedding_csv():
    word_embedding = get_word_embedding()
    with open('data/data_embeddings.csv', 'w') as f:
        # create the csv writer
        writer = csv.writer(f)
        for dictionary in word_embedding:
            for key, value in dictionary.items():
                writer.writerow([key, value])


def save_word_docs_csv():
    merge_value_positive = merge_value(list_of_sample_id_positive()) # Words appear in documents (d_pos)

    with open('C:/Users/salbo/puthineath/eurovoc_conversion/eurovoc_conversion/data/word_docs_pos.csv','w') as f:
        writer = csv.writer(f)
        for i in merge_value_positive:
            for word,docs_pos_list in i.items():
                writer.writerow([word, docs_pos_list])

    merge_value_negative = merge_value(list_of_sample_id_negative()) # Words do not appear in documents (d_neg)
    with open('C:/Users/salbo/puthineath/eurovoc_conversion/eurovoc_conversion/data/word_docs_neg.csv','w') as f:
        writer = csv.writer(f)
        for i in merge_value_negative:
            for word,docs_neg_list in i.items():
                writer.writerow([word, docs_neg_list])
    return

def save_word_docs_txt():
    merge_value_positive = merge_value(list_of_sample_id_positive()) # Words appear in documents (d_pos)

    with open('C:/Users/salbo/puthineath/eurovoc_conversion/eurovoc_conversion/data/word_docs_pos.txt','w') as f:

        for i in merge_value_positive:
            for word,docs_pos_list in i.items():
                f.write(f"{word} {docs_pos_list}\n")

    merge_value_negative = merge_value(list_of_sample_id_negative()) # Words do not appear in documents (d_neg)
    with open('C:/Users/salbo/puthineath/eurovoc_conversion/eurovoc_conversion/data/word_docs_neg.txt','w') as f:

        for i in merge_value_negative:
            for word,docs_neg_list in i.items():
                f.write(f"{word} {docs_neg_list}\n")
    return

def w2_training():
    shape = (1, 300)
    le = torch.rand(shape) # hidden layer 1 x 300
    shape = (1, 5)
    M = torch.zeros(shape)
    K = 5
    shape = (300, K)
    w2 = torch.rand(shape)
    w2_t = torch.transpose(w2, 0, 1)
    intermediate = le * w2
    le_prime = torch.sigmoid(intermediate)
    #le_prime = torch.sigmoid()
    print(le)
    return



def main():
    print(f'Words appear in documents (d_pos):\n{merge_value(list_of_sample_id_positive())}\n')
    print(f'Words do not appear in documents (d_neg):\n{merge_value(list_of_sample_id_negative())}')
    # save_txt() # already saved in the data folder
    # doc_dict_list = doc_id()
    # save_clean_doc_id_csv(doc_dict_list)
    # save_embedding_csv()
    # save_clean_doc_id_txt(doc_dict_list)
    # print(f'Words appear in documents (d_pos):\n{merge_value(list_of_sample_id_positive())}\n')
    # print(f'Words do not appear in documents (d_neg):\n{list_of_sample_id_negative()}')
    # print(f'Words appear in documents (d_pos):\n{list_of_sample_id_positive()}')
    return
#test

if __name__ == '__main__':
    # main()
    # print(get_word_embedding()
    # print(word2vec())
    print(save_word_docs_txt())






