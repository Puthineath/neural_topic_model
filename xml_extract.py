# extract data from XML files
import csv
import glob
from bs4 import BeautifulSoup as bs
import spacy
from pymongo import MongoClient
from pprint import pprint

nlp = spacy.load('en_core_web_sm')
content = []
result_list = []
folders = ["CELEX - 32017R1004", "CELEX - 32019R0833", "CELEX - 32013R1303", "CELEX - 32019R0473"]
tag_names = ["ti.art", "title", "sti.art"]

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
    files = glob.glob('data/' + folder + '/*.xml')
    # merge
    with open("data/" + folder + "/concat.xml", "wb") as outfile:
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
            data[key] = pre_processing(value)

        # add label
        data['label'] = get_data_mongo()
        return data

# Get the URI from the "theme.csv" file
def read_theme_file(folder):
    list_uri = []
    with open('data/' + folder +'/theme.csv', 'r') as file:
    # with open('data/CELEX - 32017R1004/theme.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            list_uri.append(row)
            # print(row)
    del list_uri[0]
    return list_uri

# Get the label from the MongoDB
def get_data_mongo():
    label_list = []
    myclient = MongoClient("mongodb://localhost:27017")
    db = myclient["Eurovoc"]
    collection = db["data_eu"]
    try:
        for folder in folders:
            for i in read_theme_file(folder):
                for uri in i:
                    # label_list.append(collection.find_one({"_id": uri}, {"path":0, "_id":0,"pref_label":1,"alt_label":1,"definition":1}))
                    # label_list.append(collection.find_one({"_id": uri},{"path":0,"_id":0}))

                    # Get only the 'labels', not URI or path
                    for key,value in collection.find_one({"_id": uri},{"path":0,"_id":0}).items():
                        label_list.append(value)
                        # label_list.append("".join(value))
    except:
        pass
    return label_list


def main():
    for folder in folders:
        # merge all files in each folder
        merge_files(folder)
        # put the all documents to the list
        result_list.append(get_data("data/" + folder + "/concat.xml"))

    # save to csv file
    csv_columns = ['title', 'article','label']
    csv_file = "data/data1.csv"
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in result_list:
                writer.writerow(data)
    except IOError as e:
        print(e)


if __name__ == '__main__':
    # ***** need to get the label one by one title because current code merges all the label in one
    # ***** need to get convert some labels in the list into the string
    main()
    # pprint(get_data_mongo())
    # print(read_theme_file(folders[0]))
    # print(get_data("data/CELEX - 32017R1004/concat.xml"))