from gensim.models import KeyedVectors
def load_word(word):
    # Do not forget to download the datasets and put in the data folder
    data_path = "../data/GoogleNews-vectors-negative300.bin.gz"
    # Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
    model_embeddings = KeyedVectors.load_word2vec_format(data_path, binary=True)
    try:
        embedding = model_embeddings[word]
    except:
        # if word not in pre_trained model, pad with 0 value
        embedding = 0
    return embedding

if __name__ == '__main__':
    print(load_word('dog'))
