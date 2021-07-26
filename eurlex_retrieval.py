# This script retrieve the legal document and their metadata based on a specific Tag list
# @param : tag_list & num_of_file
#

import sys
from SPARQLWrapper import SPARQLWrapper, JSON
from pprint import pprint
from pymongo import MongoClient
import spacy


tag_list = ["social policy", "fishery"]

num_of_file = 20

endpoint_url = "http://publications.europa.eu/webapi/rdf/sparql"

concept_scheme = "http://eurovoc.europa.eu/100141"

#nlp = spacy.load('en_core_web_sm')

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
            FILTER regex(?value, ".*#word.*", "i")
            }

"""
# query to retrieve an Eurlex document based on its Celex number


query_based_on_celex = """


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

# Query to retrieve an EurLex document based on the Eurovoc tag


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

data_query = """
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX euvoc: <http://publications.europa.eu/ontology/euvoc#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?prefLabel ?altLabel ?definition 
            FROM <#scheme>
            WHERE {
                optional {
                    <#concept> skos:altLabel ?altLabel .
                    filter (lang(?altLabel) = "en") 
                }

                optional { 
                    <#concept> euvoc:xlDefinition ?def .
                    ?definition rdf:value ?definition.
                    filter (lang(?definition) = "en") 
                }    
                optional {
                    <#concept> skos:prefLabel ?prefLabel .
                    filter (lang(?prefLabel) = "en") 
                }
            }
        """


def get_results(endpoint, query):
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

# def get_results_query_document_tags(endpoint):
#     "?s cdm: work_is_about_concept_eurovoc ?eurovoc_uri.
#     return assertion


def retrieve_concept_uri():
    concept_uris = []
    for elt in tag_list:
        new_query = query_tag_to_concept.replace("#word", elt)
        new_query = new_query.replace("#ConceptScheme", concept_scheme)
        lst = get_results(endpoint_url, new_query)
        for elt in lst['results']['bindings']:
            concept_uris.append(elt['c']['value'])
    return concept_uris


def retrieve_document():
    concept_list = retrieve_concept_uri()
    #document_list = get_results_query_document_tags(endpoint_url, new_query)
    #return document_list
    return concept_list


def main():
    print("------------Get the XML document------------------")
    document_list = retrieve_document()

    return


if __name__ == '__main__':
    main()