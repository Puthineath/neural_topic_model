"""
    - Get the unique tokens from datasets
    - Match the word in dataset to the embedding from word2vec
    - Get the dictionary {word:doc} of words whether it appear in each document
"""
# nltk.download('wordnet')
import csv
import pickle
import pandas as pd
from pre_processing import pre_processing
import gensim
from gensim import corpora

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
    with open('../data/clean_docs.csv', 'w') as f:
        # create the csv writer
        writer = csv.writer(f)
        for dictionary in doc_dict_list:
            for key, value in dictionary.items():
                writer.writerow([key, value])
        return


def save_clean_doc_id_txt(doc_dict_list):
    with open('../data/clean_docs.txt', 'w') as f:
        # create the csv writer
        for dictionary in doc_dict_list:
            for id, tokens_list in dictionary.items():
                f.write(f"{id} {tokens_list}\n")
        return


# check if the a word in the document or not
# doc positive
"""
    form the word and doc : {word:doc}
"""

def sample_doc_positive(word):
    word_in_doc_list = []
    for i in doc_id():
        for key, value in i.items():
            if word in value:
                word_in_doc_list.append({word:key})

    return word_in_doc_list
    # return word_in_doc_list,word_not_in_doc_list

# doc negative
def sample_doc_negative(word):
    word_not_in_doc_list = []
    for i in doc_id():

        for key, value in i.items():
            if word not in value:
                word_not_in_doc_dict = {word:key}
                word_not_in_doc_list.append(word_not_in_doc_dict)
            else:
                word_not_in_doc_list.append({word: None})
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
    new_dict1={}
    for d in dict_list:
        for d_key,d_value in d.items():
            if d_key not in new_dict:
                new_dict[d_key] = []
            new_dict[d_key].append(d[d_key])
    for k,v in new_dict.items():
        new_dict1[k] = [val for val in v if val != None]
    return new_dict1


def merge_value(list_of_sample_id):
    merge_value_list=[]
    for dict in list_of_sample_id:
        merge_value_list.append(merge_list_of_dictionaries(dict))
    return  merge_value_list

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


if __name__ == '__main__':
    main()
    # print(get_word_embedding()
    # print(word2vec())
    # print(save_word_docs_txt())






