import sys
from SPARQLWrapper import SPARQLWrapper, JSON
from pprint import pprint

endpoint_url = "http://localhost:3030/Human-sex/sparql"

# get the top concept
query2 = """
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            SELECT ?concept
            FROM <http://www.data.com/3>
            WHERE {
                ?concept skos:topConceptOf ?conceptScheme  .
            }
            # LIMIT 5
        """
# get narrower concept
query_narrow = """
                    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
                    PREFIX euvoc: <http://publications.europa.eu/ontology/euvoc#>
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
                    SELECT ?child
                    FROM <http://www.data.com/3>
                    WHERE {
                        <%(concept)s> skos:narrower ?child  .
                    }
                    # LIMIT 5
                """
def get_results(endpoint_url, query,concept = ""):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    # sparql.setQuery(query)
    sparql.setQuery(query % {'concept': str(concept)})
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def get_top_concept(results):
    top_concept = [result['concept']['value'] for result in results["results"]["bindings"] ]
    return top_concept

def get_narrow(top_concept):
    narrow_results = get_results(endpoint_url, query_narrow, concept=top_concept)
    narrow_results_list = [narrow_result['child']['value'] for narrow_result in narrow_results["results"]["bindings"]]
    for elt in narrow_results_list:
        print(elt)
        get_narrow(elt)
    return narrow_results_list

# def get_narrower(top_concepts):
#     for top_concept in top_concepts:
#         narrow_results = get_results(endpoint_url, query2, concept=top_concept)
#         narrow_results_list = [narrow_result['concept']['value'] for narrow_result in narrow_results["results"]["bindings"]]
#         for each_narrow in narrow_results_list:
#             narrow_results = get_results(endpoint_url, query2, concept=top_concept)
#             narrow_results_list = [narrow_result['concept']['value'] for narrow_result in narrow_results["results"]["bindings"]]
#     return narrow_results_list


# def get_narrower(top_concepts):
#     for top_concept in top_concepts:
#         narrow_results = get_results(endpoint_url, query2, concept=top_concept)
#         narrow_results_list = [narrow_result for narrow_result in narrow_results["results"]["bindings"]]
#     return narrow_results_list

# def recursive_narrower():



def main():
    print("------------Get the top concepts------------------")
    results = get_results(endpoint_url, query2, concept="?concept")
    top_concepts = get_top_concept(results)
    # pprint(top_concepts)
    print("------------Get narrower concepts------------------")

    lis = [get_narrow(top_concept) for top_concept in top_concepts]
    pprint(lis)
    print(len(lis))




if __name__ == '__main__':
    pprint(main())