 # extract data from XML files
import csv
import glob
import os
from bs4 import BeautifulSoup as bs
import spacy
from pprint import pprint

nlp = spacy.load('en_core_web_sm')
content = []
result_list = []
folders = ["CELEX - 32017R1004","CELEX - 32019R0833","CELEX - 32013R1303","CELEX - 32019R0473"]
tag_names = ["ti.art","title","sti.art"]

# ***** need to extract the labels from MongoDB

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

def merge_files(folder):
    # get all xml files in each folder
    files = glob.glob('data/'+folder+'/*.xml')
    # merge
    with open("data/"+folder+"/concat.xml", "wb") as outfile:
        for f in files:
            with open(f, "rb") as infile:
                outfile.write(infile.read())
    return

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
        try:
            for tag_name in tag_names:
                for tag in bs_content.find("enacting.terms")(tag_name):
                    tag.decompose()
        except:
            pass

        # get the article
        data['article'] = bs_content.find("enacting.terms").get_text()

        # clean the texts
        for key, value in data.items():
            data[key]= pre_processing(value)
        return data

def main():
    for folder in folders:
        # merge all files in each folder
        merge_files(folder)
        # put the all documents to the list
        result_list.append(get_data("data/"+folder+"/concat.xml"))

    # save to csv file
    a_file = open("data/data.csv", "w")
    for result in result_list:
        writer = csv.writer(a_file)
        for key, value in result.items():
            writer.writerow([key, value])

    a_file.close()

if __name__ == '__main__':
    main()
