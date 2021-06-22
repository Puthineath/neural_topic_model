import pymongo
import sys
from SPARQLWrapper import SPARQLWrapper, JSON
#import spacy
from string import punctuation
from pymongo import MongoClient


endpoint_url = "http://localhost:3030/Human-sex/sparql"
# endpoint_url = "http://publications.europa.eu/resource/dataset/eurovoc/sparql"
# extract the definition of the word :  "EU financing"@en
query = """
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            SELECT ?label
            FROM <http://www.data.com/3>
            WHERE {
            ?label skos:prefLabel "ethics"@en .
            }
            LIMIT 25
        """

def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

def main():

    results = get_results(endpoint_url, query)
    result_list = []
    for result in results["results"]["bindings"]:
        result_list.append(result)
    myclient = MongoClient("mongodb://82.165.108.31:8081/")
    # myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    # mydb = myclient["mydatabase"]
    # mycol = mydb["customers"]
    db = myclient["Test"]
    collection = db["test"]

    # mydict = { "name": "John", "address": "Highway 37" }
    print(result_list)
    x = collection.insert_one(result_list[0])

if __name__ == '__main__':
    main()