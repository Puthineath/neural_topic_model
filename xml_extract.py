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

def pre_processing(texts):
    # remove stop words
    sw_spacy = nlp.Defaults.stop_words
    words = [word for word in texts.split() if word.lower() not in sw_spacy]
    new_text = " ".join(words)
    # get keywords
    doc = nlp(new_text)
    a = [chunk.text for chunk in doc.noun_chunks]
    # ***** need to remove the word "article" + number and remove word annex
    # ***** need to remove punctutation
    return " ".join(a)

# merge xml files
def merge_files(folder):
    # get all xml files in each folder
    files = glob.glob('data/' + folder + '/*.xml')
    # merge and save to "concat.xml" file
    with open("data/" + folder + "/concat.xml", "wb") as outfile:
        for f in files:
            with open(f, "rb") as infile:
                outfile.write(infile.read())
    return

# get titles, articles, and labels
def get_data(folder):
    data = {}
    with open("data/" + folder + "/concat.xml", "r", encoding="utf-8") as xml_file:
        # Read each line in the file, readlines() returns a list of lines
        content = xml_file.readlines()

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
        # ** data['label'] = get_data_mongo(folder)

        # get only one label for testing
        data['label'] = get_data_mongo(folder)[0]

        return data


# Get the URI from the "theme.csv" file
def read_theme_file(folder):
    list_uri = []
    with open('data/' + folder + '/theme.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            list_uri.append(row)
            # print(row)
    del list_uri[0]
    return list_uri


# Get the labels from the MongoDB
def get_data_mongo(folder):
    label_list = []
    myclient = MongoClient("mongodb://localhost:27017")
    db = myclient["Eurovoc"]
    collection = db["data_eu"]

    try:
        for list_uri in read_theme_file(folder):
            for uri in list_uri:
                # Get only the 'labels', neither URI nor path
                for key, value in collection.find_one({"_id": uri}, {"path": 0, "_id": 0}).items():
                    if type(value) == str:
                        label_list.append(value)

                    elif type(value) == list:
                        for i in value:
                            label_list.append("".join(i))
    except:
        pass
    # remove empty strings
    label_list = [x for x in label_list if x != '']
    return label_list


def main():
    for folder in folders:
        # merge all files in each folder
        merge_files(folder)
        # put the all documents to the list
        result_list.append(get_data(folder))

    # save to csv file
    csv_columns = ['title', 'article', 'label']
    csv_file = "data/data1.csv"
    # csv_file = "data/data_prefLabel.csv"
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in result_list:
                writer.writerow(data)
    except IOError as e:
        print(e)

if __name__ == '__main__':
    main()

