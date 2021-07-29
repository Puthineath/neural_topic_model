# This script retrieve the legal document and their metadata based on a specific Tag list
# @param : tag_list & num_of_file
#

import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import requests
from pprint import pprint
from pymongo import MongoClient
import spacy


tag_list = ["fishery"]

num_of_file = 20

endpoint_url = "http://publications.europa.eu/webapi/rdf/sparql"

concept_scheme = "http://eurovoc.europa.eu/100141"

# Query to retrieve the concept with the prefLabel from the tag_list
query_tag_to_concept = """ 
            Prefix cdm: <http://publications.europa.eu/ontology/cdm#>
            Prefix skos: <http://www.w3.org/2004/02/skos/core#>
            Prefix skosxl: <http://www.w3.org/2008/05/skos-xl#>
            
            select ?c  from <#ConceptScheme> where  { 
            
            ?c a skos:Concept.
            ?c skosxl:prefLabel ?lab .
            ?lab skosxl:literalForm ?value.
            filter(lang(?value) = "en").
            FILTER regex(?value, "#word", "i")
            }

"""

query_based_on_tag = """
prefix cdm: <http://publications.europa.eu/ontology/cdm#>

select ?celex_id  where {
            #assertion         
            ?s cdm:work_id_document ?celex_id .
            ?s ^cdm:expression_belongs_to_work ?expression .    
            ?expression cdm:expression_uses_language <http://publications.europa.eu/resource/authority/language/ENG> .
            ?expression ^cdm:manifestation_manifests_expression ?manifestation .
            ?manifestation ^cdm:item_belongs_to_manifestation ?item .
            ?manifestation cdm:manifestation_type ?manifestationtype .
        filter ( str(?manifestationtype) ='fmx4')

} limit 100


"""

# Query to retrieve the Eurovoc concept link to a specific work document
query_document_tags = """ prefix cdm: <http://publications.europa.eu/ontology/cdm#>

select ?eurovoc_uri  where {
            ?s cdm:work_is_about_concept_eurovoc ?eurovoc_uri .
            ?s cdm:work_id_document "celex:#celex"^^<http://www.w3.org/2001/XMLSchema#string> .
            ?s ^cdm:expression_belongs_to_work ?expression .    
            ?expression cdm:expression_uses_language <http://publications.europa.eu/resource/authority/language/ENG> .
            ?expression ^cdm:manifestation_manifests_expression ?manifestation .
            ?manifestation ^cdm:item_belongs_to_manifestation ?item .
            ?manifestation cdm:manifestation_type ?manifestationtype .
        filter ( str(?manifestationtype) ='fmx4')

}  """

# Query to retrieve an EurLex document based on the E
query_doc_from_celex = """
        prefix cdm: <http://publications.europa.eu/ontology/cdm#>

        select * where {
        #?s cdm:work_id_document "celex:32015R1929"^^<http://www.w3.org/2001/XMLSchema#string> .
        ?s cdm:work_id_document "#celex"^^<http://www.w3.org/2001/XMLSchema#string> .
        ?s ^cdm:expression_belongs_to_work ?expression.
        ?expression cdm:expression_uses_language <http://publications.europa.eu/resource/authority/language/ENG>.
        ?expression ^ cdm:manifestation_manifests_expression ?manifestation.
        ?manifestation ^ cdm:item_belongs_to_manifestation ?item.
        ?manifestation cdm:manifestation_type ?manifestationtype.
        filter(str(?manifestationtype) ='fmx4')
        } limit 100         """

def get_results(endpoint, query):
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def get_results_query_document_tags(endpoint, concept):
    # Retrieve the documents based on the URI of the tag - rsults : list of celex and oj documents
    assertion = "?s cdm:work_is_about_concept_eurovoc <#eurovoc_uri> . "
    global_assert = assertion.replace("#eurovoc_uri", concept)
    new_query = query_based_on_tag.replace("#assertion", global_assert)
    result_list = get_results(endpoint_url, new_query)
    # Extract the documents found from the Cellar
    for doc in result_list['results']['bindings']:
        if "celex:" in doc['celex_id']['value']:
            new_query = query_doc_from_celex.replace("#celex", doc['celex_id']['value'])
            doc_parts = get_results(endpoint_url, new_query)
            for part in doc_parts['results']['bindings']:
                r = requests.get(part['item']['value'])
                push_to_mongodb("")
            print(new_query)
    return assertion


def push_to_mongodb(part):
    return


def retrieve_concept_uri():
    concept_uris = []
    for elt in tag_list:
        new_query = query_tag_to_concept.replace("#word", elt)
        new_query = new_query.replace("#ConceptScheme", concept_scheme)
        lst = get_results(endpoint_url, new_query)
        for elt in lst['results']['bindings']:
            concept_uris.append(elt['c']['value'])
    return concept_uris


def retrieve_document(concept_list):
    document_list = []
    for concept in concept_list:
        document_list.append(get_results_query_document_tags(endpoint_url, concept))
        if len(document_list) > num_of_file :
            return document_list
    return document_list


def main():
    print("------------Get the XML document------------------")
    concept_list = retrieve_concept_uri()
    document_list = retrieve_document(concept_list)
    return


if __name__ == '__main__':
    main()
