import sys
from SPARQLWrapper import SPARQLWrapper, JSON
from pprint import pprint
import pymongo

endpoint_url = "http://localhost:3030/Human-sex/sparql"

# get the top concept
query2 = """
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            SELECT ?concept
            FROM <http://www.data.com/3>
            WHERE {
                ?concept skos:topConceptOf ?conceptScheme  .
            }
            limit 10
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
                    limit 10
                """
# get definition and preflabel
query_def = """
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX euvoc: <http://publications.europa.eu/ontology/euvoc#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?Definition ?label ?child_id
            FROM <http://www.data.com/3>
            WHERE {
                <%(concept)s> euvoc:xlDefinition ?label .
                ?label rdf:value ?Definition.
                <%(concept)s> skos:prefLabel "EU financing"@en .
                <%(concept)s> skos:narrower ?child_id  .
            }
            limit 10
        """

def get_results(endpoint_url, query, concept=""):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    # sparql.setQuery(query)
    sparql.setQuery(query % {'concept': str(concept)})
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def get_top_concept(results):
    top_concept = [result['concept']['value'] for result in results["results"]["bindings"]]
    return top_concept

def get_narrow(top_concept):
    # query definition, label, narrower concept
    narrow_results = get_results(endpoint_url, query_def, concept=top_concept)
    narrow_results_list = [narrow_result for narrow_result in narrow_results["results"]["bindings"]]
    for elt in narrow_results_list:
        # print(elt)
        # get only URI to continue querying
        get_narrow(elt['child_id']['value'])

    return narrow_results_list


def main():
    store = []
    # print("------------Get the top concepts------------------")
    results = get_results(endpoint_url, query2, concept="?concept")
    top_concepts = get_top_concept(results)
    # pprint(top_concepts)
    print("------------Get narrower concepts------------------")

    # for l in lises:
    #     store.extend(l)
    # print(len(store))
    lists = [get_narrow(top_concept) for top_concept in top_concepts]
    pprint(lists)
    print(len(lists))
    for l in lists:
        store.extend(l)
    print(len(store))



if __name__ == '__main__':
    # pprint(main())
    main()