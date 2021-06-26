import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import spacy
from pprint import  pprint
from string import punctuation

nlp = spacy.load('en_core_web_sm')
endpoint_url = "http://localhost:3030/Human-sex/sparql"
# extract the definition of the word :  "EU financing"@en
query = """
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX euvoc: <http://publications.europa.eu/ontology/euvoc#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?definition ?value ?concept
            FROM <http://www.data.com/3>
            WHERE {
                ?concept euvoc:xlDefinition ?definition .
                ?definition rdf:value ?value.
                ?concept skos:prefLabel "EU financing"@en .
            }
            LIMIT 25
        """
# query1 = """
#     PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
#     PREFIX euvoc: <http://publications.europa.eu/ontology/euvoc#>
#     PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
#
#     SELECT ?value ?definition ?label
#     FROM <http://www.data.com/3>
#     WHERE {
#
#       <http://eurovoc.europa.eu/1005> euvoc:xlDefinition ?definition .
#       ?definition rdf:value ?value.
#       <http://eurovoc.europa.eu/1005> skos:prefLabel ?label .
#       filter(lang(?definition) = "en")
#     }
#     LIMIT 25
# """
def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    # sparql.setQuery(query % {'concept': str(concept)})
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

def preprocessing(texts):
    # remove stop words
    sw_spacy = nlp.Defaults.stop_words
    words = [word for word in texts.split() if word.lower() not in sw_spacy]
    new_text = " ".join(words)
    # get keywords
    doc = nlp(new_text)
    # need to convert to string, not list
    a = [chunk.text for chunk in doc.noun_chunks]
    return a

def main():

    list_of_definition = []
    results = get_results(endpoint_url, query)
    # get keywords from each dictionary
    for result in results["results"]["bindings"]:
        result['value']['value'] = preprocessing(result['value']['value'])
        # print(result)
    pprint(results)
    # convert_definition(list)


if __name__ == "__main__":
    # print(main())
    main()

