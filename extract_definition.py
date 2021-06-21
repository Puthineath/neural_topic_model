import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import spacy
from string import punctuation

nlp = spacy.load('en_core_web_sm')
endpoint_url = "http://localhost:3030/Human-sex/sparql"
# extract the definition of the word :  "EU financing"@en
query = """
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX euvoc: <http://publications.europa.eu/ontology/euvoc#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?Definition
            FROM <http://www.data.com/3>
            WHERE {
                ?c euvoc:xlDefinition ?a .
                ?a rdf:value ?Definition.
                ?c skos:prefLabel "EU financing"@en .
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

# def get_hotwords(text):
# #     result = []
# #     pos_tag = ['PROPN', 'ADJ', 'NOUN']  # 1
# #     doc = nlp(text.lower())  # 2
# #     for token in doc:
# #         # remove stop word and punctuation
# #         if (token.text in nlp.Defaults.stop_words or token.text in punctuation):
# #             continue
# #         # keep only pronounce, adjective, and noun
# #         if (token.pos_ in pos_tag):
# #             result.append(token.text)
# #     return result
def get_hotwords(texts):
    doc = nlp(texts)
    a = [chunk.text for chunk in doc.noun_chunks]
    return a

def main():
    list_of_definition = []
    list_of_lang = []
    results = get_results(endpoint_url, query)
    for result in results["results"]["bindings"]:
        # get the definition for each language
        list_of_definition.append(result['Definition']['value'])
        #
        list_of_lang.append(result['Definition']['xml:lang'])
    dic_definiton = dict(zip(list_of_lang, list_of_definition))
    for key, value in dic_definiton.items():
        print("{}:{}".format(key, get_hotwords(value)))

if __name__ == "__main__":
    main()

