# Get the labels and push to MongoDB

import sys
from urllib.error import HTTPError

from SPARQLWrapper import SPARQLWrapper, JSON
from pprint import pprint
from pymongo import MongoClient
import spacy
import time

#endpoint_url = "http://publications.europa.eu/webapi/rdf/sparql"
endpoint_url = "http://localhost:3030/eurovoc-load/query"
concept_scheme = "http://eurovoc.europa.eu/100141"


nlp = spacy.load('en_core_web_sm')

# get the top concept
query_top_concept = """
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            SELECT #concept
            FROM <#scheme>
            WHERE {
                #concept skos:topConceptOf <#scheme>  .
            }
        """
# get narrower concept
query_narrow = """
                    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
                    PREFIX euvoc: <http://publications.europa.eu/ontology/euvoc#>
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

                    SELECT ?child
                    FROM <#scheme>
                    WHERE {
                        <#concept> skos:narrower ?child  .
                    }
                """

preflabel_query = """
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX euvoc: <http://publications.europa.eu/ontology/euvoc#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?prefLabel 
            FROM <#scheme>
            WHERE {
                    <#concept> skos:prefLabel ?prefLabel .
                    filter (lang(?prefLabel) = "en") 
            }
        """

altLabel_query = """
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX euvoc: <http://publications.europa.eu/ontology/euvoc#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?altLabel  
            FROM <#scheme>
            WHERE {
                    <#concept> skos:altLabel ?altLabel .
                    filter (lang(?altLabel) = "en")   
            }
"""

definition_query = """
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX euvoc: <http://publications.europa.eu/ontology/euvoc#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?definition  
            FROM <#scheme>
            WHERE {
                    <#concept> euvoc:xlDefinition ?def .
                    ?def rdf:value ?definition.
                    filter (lang(?definition) = "en")  
            }
"""


def get_results(endpoint, query, concept=""):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(endpoint, agent=user_agent)
    tmp_query = query.replace("#concept", concept)
    tmp_query = tmp_query.replace("#scheme", concept_scheme)
    sparql.setQuery(tmp_query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def get_top_concept(results):
    top_concept = [result['concept']['value'] for result in results["results"]["bindings"]]
    return top_concept


def push_mongo_db(uri, pref_label, alt_label, definition, path):
    myclient = MongoClient("mongodb://localhost:27017")
    db = myclient["Eurovoc2"]
    collection = db["data_eu"]

    try:
        if collection.find_one({"_id": uri}):
            # delete and replace the duplicated id
            collection.delete_one({"_id": uri})
            item_1 = {
                "_id": uri,
                "pref_label": pref_label,
                "alt_label": alt_label,
                "definition": definition,
                "path": path
            }
            # insert data to MongoDB
            collection.insert_many([item_1])
            # count the number in database to sleep 60 seconds every 100 entries of documents
            if collection.count_documents({}) % 100 == 0:
                print(f"insert{collection.count_documents({})}")
        else:
            item_1 = {
                "_id": uri,
                "pref_label": pref_label,
                "alt_label": alt_label,
                "definition": definition,
                "path": path
            }
            collection.insert_many([item_1])
            # count the number in database to sleep 60 seconds every 100 entries of documents
            if collection.count_documents({}) % 100 == 0:
                print(f"insert{collection.count_documents({})}")

    except HTTPError as e:
        print(e)

    return


def pre_processing(texts):
    # remove stop words
    sw_spacy = nlp.Defaults.stop_words
    words = [word for word in texts.split() if word.lower() not in sw_spacy]
    new_text = " ".join(words)
    # get keywords
    doc = nlp(new_text)
    # need to convert to string, not list
    a = [chunk.text for chunk in doc.noun_chunks]
    return a


def get_narrow(concept, path):
    # Retrieve the information about the concept
    data_results = get_results(endpoint_url, " ".join(preflabel_query.split()), concept=concept)
    pref_label = data_results['results']['bindings'][0]['prefLabel']['value']
    path = path + "," + "<" + concept + ">"
    if 'altLabel' in data_results['results']['bindings'][0]:
        alt_label = data_results['results']['bindings'][0]['altLabel']['value']
    else:
        alt_label = ""
    if 'definition' in data_results['results']['bindings'][0]:
        definition = data_results['results']['bindings'][0]['definition']['value']
        def_wording = pre_processing(definition)
        definition = def_wording
    else:
        definition = ""

    # Push the data to MongoDB
    push_mongo_db(concept, pref_label, alt_label, definition, path)

    # Retrieve the narrower concepts if any
    narrow_results = get_results(endpoint_url, " ".join(query_narrow.split()), concept=concept)
    narrow_results_list = [narrow_result['child']['value'] for narrow_result in narrow_results["results"]["bindings"]]
    for elt in narrow_results_list:
        get_narrow(elt, path)
    return narrow_results_list


def main():
    print("------------Get the top concepts------------------")
    results = get_results(endpoint_url, " ".join(query_top_concept.split()), concept="?concept")
    # pprint(results)
    top_concepts = get_top_concept(results)
    pprint(top_concepts)
    print(len(top_concepts))
    print("------------Get narrower concepts------------------")
    lis = [get_narrow(top_concept, "<" + concept_scheme + ">") for top_concept in top_concepts]

    return


if __name__ == '__main__':
    print(main())
