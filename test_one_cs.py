# test on only one concept scheme "http://eurovoc.europa.eu/100239"

import sys
from SPARQLWrapper import SPARQLWrapper, JSON
from pprint import pprint


endpoint_url = "http://localhost:3030/Human-sex/sparql"
# query to get top concepts
query_concept="""
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX euvoc: <http://publications.europa.eu/ontology/euvoc#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?child
FROM <http://www.data.com/3>
WHERE {
?child skos:topConceptOf <%(concept)s>  .
}
"""

# query to get definition, label, and value
query_def2 = """
PREFIX skosxl: <http://www.w3.org/2008/05/skos-xl#>

SELECT  ?definition ?value ?label ?newval ?child
FROM <http://www.data.com/3>
WHERE {
  <%(concept)s> euvoc:xlDefinition ?definition .
  ?definition rdf:value ?value .
   <%(concept)s> skosxl:prefLabel ?label .
  ?label skosxl:literalForm ?newval . 
  <%(concept)s> skos:narrower ?child .
  filter(lang(?newval) = "en")
  
}
}
"""
# query to get narrower concept
query_narrow = """
                    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
                    PREFIX euvoc: <http://publications.europa.eu/ontology/euvoc#>
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

                    SELECT ?child
                    FROM <http://www.data.com/3>
                    WHERE {
                        <%(concept)s> skos:narrower ?child  .
                    }
                    
                """

def get_results(endpoint_url, query, concept=""):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    # sparql.setQuery(query)
    sparql.setQuery(query % {'concept': str(concept)})
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

# function to get top concepts URI
def get_top_concept(results):
    top_concept = [result['child']['value'] for result in results["results"]["bindings"]]
    return top_concept

# function to get narrower concepts URI
def get_narrow(top_concept):
    lis = []
    # lis=[top_concept]
    lis.append(top_concept)
    narrow_results = get_results(endpoint_url, query_narrow, concept=top_concept)
    narrow_results_list = [narrow_result['child']['value'] for narrow_result in narrow_results["results"]["bindings"]]
    # lis = [get_narrow(elt) for elt in narrow_results_list]
    for elt in narrow_results_list:

        lis.append(get_narrow(elt))
        # get_narrow(elt)

    return lis
    # return narrow_results_list


def main():
    store = []
    # print("------------Get the top concepts------------------")
    results = get_results(endpoint_url, query_concept, concept="http://eurovoc.europa.eu/100239")
    top_concepts = get_top_concept(results)

    # get the narrower concepts
    for top_concept in top_concepts:
        pprint(get_narrow(top_concept))


if __name__ == '__main__':
    # pprint(main())
    main()