# pre_process data
from gensim.models import KeyedVectors
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import re
from spacy.lang.en import English
from nltk.corpus import wordnet as wn


parser = English()

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

