# pre_process data

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

if __name__ == '__main__':
    text = '(EU 17 2017 establishment Union framework collection management use data fisheries sector Council Regulation (EC,"1.With view (EU Regulation management biological, environmental, technical socioeconomic data fisheries sector Article Regulation (EU 1380/2013.2.The data paragraph shall collected obligation collect Union legal acts Regulation.3.For data necessary fisheries management Union legal acts Regulation rules transmission data , processing, management use data collected Regulation (EC (EC 223/2009.For purposes Regulation referred Article 4 Regulation (EU addition definitions sector activities related commercial fisheries recreational fisheries aquaculture industries processing fisheries products'
    print(pre_processing(text))
