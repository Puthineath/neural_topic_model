import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import spacy
from string import punctuation

endpoint_url = "http://localhost:3030/Human-sex/sparql"

# get the concept scheme
query = """
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            SELECT (SAMPLE(?conceptScheme) AS ?scheme)
            FROM <http://www.data.com/3>
            WHERE {
                ?concept skos:topConceptOf ?conceptScheme  .
            }
        """

# get the child of the top Concept
query1 = """
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX euvoc: <http://publications.europa.eu/ontology/euvoc#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?child
            #FROM <http://eurovoc.europa.eu/100141>
            FROM <http://www.data.com/3>
            WHERE {
                ?concept skos:narrower ?child  .
            }
            LIMIT 10
        """
# get the top concept
query2 = """
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            SELECT ?concept
            FROM <http://www.data.com/3>
            WHERE {
                ?concept skos:topConceptOf ?conceptScheme  .
            }
            LIMIT 10
        """

def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

def get_each_result(results):
    for result in results["results"]["bindings"]:
        print(result)

print("------------Get the concept scheme------------------")
results = get_results(endpoint_url, query)
get_each_result(results)

print("-------------Get the top concept---------------------")

results2 = get_results(endpoint_url, query2)
get_each_result(results2)

# need to create func to call only once . recall yourself each time calling narrower concept for recursion.
# dont need to recall many time
# to find the distance between the graph

print("------------Get the child of top concept----------------------")
results1 = get_results(endpoint_url, query1)
get_each_result(results1)




