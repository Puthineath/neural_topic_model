from bs4 import BeautifulSoup as bs
import spacy
from pprint import pprint

nlp = spacy.load('en_core_web_sm')
content = []
result_list = []
num_datas = ["CELEX - 32006D0199","CELEX - 32017R1004","CELEX - 32019R0833","CELEX - 32013R1303","CELEX - 32019R0473"]
tag_names = ["ti.art","title","sti.art"]

# need to extract the label

def add_path(data):
    # ***** need to rename the doc file in the folder by removing (1),(2),...
    PATH = "data/" + data + "/DOC_2.xml"
    return PATH

def pre_processing(texts):
    # remove stop words
    sw_spacy = nlp.Defaults.stop_words
    words = [word for word in texts.split() if word.lower() not in sw_spacy]
    new_text = " ".join(words)
    # get keywords
    doc = nlp(new_text)
    a = [chunk.text for chunk in doc.noun_chunks]

    # need to remove the word "article" + number and remove word annex
    # need to remove punctutation

    return " ".join(a)

def get_data(file):
    data = {}

    with open(file, "r", encoding="utf-8") as file:

        # Read each line in the file, readlines() returns a list of lines
        content = file.readlines()

        # Combine the lines in the list into a string
        content = "".join(content)
        bs_content = bs(content, "lxml")

        # get the title
        data['title'] = bs_content.find("title").get_text()

        #  remove unimportant tag(s) to get the important info in the article
        for tag_name in tag_names:
            for tag in bs_content.find("enacting.terms")(tag_name):
                tag.decompose()

        # get the article
        data['article'] = bs_content.find("enacting.terms").get_text()

        # clean the texts
        for key, value in data.items():
            data[key]= pre_processing(value)
        return data

def main():
    for num_data in num_datas:
        result_list.append(get_data(add_path(num_data)))
    print(result_list)


if __name__ == '__main__':
    main()
