# coding=utf-8

from stanfordcorenlp import StanfordCoreNLP
from nltk.parse.dependencygraph import DependencyGraph, malt_demo
import networkx as nx
import json

def nxGraphWroot(dep_graph):
    """Convert the data in a ``nodelist`` into a networkx labeled directed graph.
        Include the ROOT node
    """
    import networkx

    nx_nodelist = list(range(0, len(dep_graph.nodes))) ##
    nx_edgelist = [
        (n, dep_graph._hd(n), dep_graph._rel(n))
        for n in nx_nodelist
    ]
    dep_graph.nx_labels = {}
    for n in nx_nodelist:
        dep_graph.nx_labels[n] = dep_graph.nodes[n]['word']

    g = networkx.MultiDiGraph()
    g.add_nodes_from(nx_nodelist)
    g.add_edges_from(nx_edgelist)

    return g

class RESyntaxer():

    def __init__(self):
        self.corenlp_server = StanfordCoreNLP('http://localhost', port=9000)

    def get_token(self, text):
        """
        :param text: a list of unprocessed sentence
        :param nlp_server: an initialized stanford corenlp server
        :return: a list of sentences information (split sentences, tokenized words, Part-of-Speech of tokens)
        """
        props = {'annotators': 'tokenize', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
        text_parse = self.corenlp_server.annotate(text, properties=props)
        text_json = json.loads(text_parse)['tokens']
        return [ tok['originalText'] for tok in text_json]

    def get_dep_graph(self,
                      sent,
                      dep_ver='SD'  ## 'SD': stanford dependency, 'UD': universal dependency
                      ):
        props={'annotators': 'tokenize, ssplit, pos, depparse',
               'pipelineLanguage': 'en',
               'outputFormat': 'json',
               'depparse.model': "edu/stanford/nlp/models/parser/nndep/english_%s.gz" % dep_ver}
        dep_parse = self.corenlp_server.annotate(sent, properties=props)
        dep_json = json.loads(dep_parse)
        dep_info_sent0 = dep_json['sentences'][0]['basicDependencies']
        tok_info_sent0 = dep_json['sentences'][0]['tokens']
        sorted_deps = sorted(dep_info_sent0, key=lambda x: x['dependent'])
        conll_str = ''.join(["%s\t%s\t%i\t%s\n" % (dep['dependentGloss'],
                                                   tok_info_sent0[dep['dependent'] - 1]['pos'],
                                                   dep['governor'],
                                                   dep['dep']) for dep in sorted_deps])

        return DependencyGraph(conll_str)

    def get_sdp(self,
                dep_graph,
                sour, ## sour token conll_id
                targ): ## targ token conll_id
        sd_dep_nx = nxGraphWroot(dep_graph).to_undirected()
        return nx.shortest_path(sd_dep_nx, source=sour, target=targ)

    def close(self):
        self.corenlp_server.close()