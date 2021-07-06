from bs4 import BeautifulSoup as bs
from pprint import pprint

content = []
# Read the XML file
def get_data(file):
    data = {}
    # data = []
    with open(file, "r") as file:
        # Read each line in the file, readlines() returns a list of lines
        content = file.readlines()
        # Combine the lines in the list into a string
        content = "".join(content)
        bs_content = bs(content, "lxml")
        title = bs_content.find("title").find("ti").find_all('p')
        # title = bs_content.find_all('p')
        # data = [i.get_text() for i in title]

        # remove the last 3 lines of unimportant title
        data['title'] = list(map(lambda x: x.get_text(), title))[:-3]
        # article = bs_content.find("enacting.terms").find_all("article")
        article = bs_content.find("enacting.terms")
        for i in article("ti.art"):
                i.decompose()
        data['article'] = list(map(lambda x: x.get_text(), article))[:-2]
        return data


def main():
    result = get_data("data/DOC_2.xml")
    pprint(result)


if __name__ == '__main__':
    main()