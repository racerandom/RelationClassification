# coding=utf-8

from stanfordcorenlp import StanfordCoreNLP
from nltk.parse.dependencygraph import DependencyGraph, malt_demo
import spacy
from spacy import displacy

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


class CorenlpParser:

    def __init__(self):
        self.corenlp_server = StanfordCoreNLP('http://localhost', port=9000)

    def tokenize(self, text):
        """
        :param text: a list of unprocessed sentence
        :param nlp_server: an initialized stanford corenlp server
        :return: a list of sentences information (split sentences, tokenized words, Part-of-Speech of tokens)
        """
        props = {'annotators': 'tokenize', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
        text_parse = self.corenlp_server.annotate(text, properties=props)
        text_json = json.loads(text_parse)['tokens']
        return [tok['originalText'] for tok in text_json]

    def get_dep_graph(self,
                      sent,
                      dep_ver='SD'  # 'SD': stanford dependency, 'UD': universal dependency
                      ):
        props={'annotators': 'pos, depparse',
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

    def visualize(self, text):
        dep_graph = self.get_dep_graph(text, dep_ver='SD')
        dep_graph.tree().draw()

    def get_SDP(self,
                text,
                sour, # sour token conll_id
                targ): # targ token conll_id
        dep_graph = self.get_dep_graph(text)
        # sd_dep_nx = nxGraphWroot(dep_graph).to_undirected()
        sd_dep_nx = dep_graph.nx_graph()
        return nx.shortest_path(sd_dep_nx, source=sour, target=targ)

    def close(self):
        self.corenlp_server.close()


def sbd_component(doc):
    for i, token in enumerate(doc[:-2]):
        # define sentence start if period + titlecase token
        # if token.text == '.' and doc[i+1].is_title:
        #     doc[i+1].sent_start = False
        if i != 0:
            doc[i].sent_start = False
    return doc


class SpacyParser:

    def __init__(self):
        self.parser = spacy.load('en')
        self.parser.add_pipe(sbd_component, before='tagger')

    def tokenize(self, text):
        tokens = self.parser(text, disable=['parser', 'tagger', 'ner'])
        return [tok.text for tok in tokens]

    def parse(self, text):
        doc = self.parser(text)
        return doc

    def visualize(self, text):
        doc = self.parser(text)
        displacy.serve(doc, style='dep')

    def get_SDP(self, text, sour_id, targ_id, returnWord=True, orderSeq=False, onlyID=True):
        try:
            doc = self.parser(text)

            # Load spacy's dependency tree into a networkx graph
            edges = []
            for token in doc:
                # FYI https://spacy.io/docs/api/token
                for child in token.children:
                    if onlyID:
                        edges.append(('%i' % token.i, '%i' % child.i))
                    else:
                        edges.append(('{0}-{1}'.format(token.lower_, token.i),
                                      '{0}-{1}'.format(child.lower_, child.i)))

            graph = nx.Graph(edges)

            sdp_ids = nx.shortest_path(graph, source=str(sour_id), target=str(targ_id))

            # print([tok.text for tok in doc])

            if orderSeq:
                sdp_ids = sorted(sdp_ids)

            if returnWord:
                return [doc[int(id)].text for id in sdp_ids]
            else:
                return sdp_ids

        except Exception as ex:
            print('[SDP ERROR] %s, %s, %s' % (text, sour_id, targ_id))
            print('[SDP ERROR] %s, %s' % (text.split()[sour_id], text.split()[targ_id]))
            print(str(ex))


# parser = SpacyParser()
#
# # The <e1>author's</e1> of a keygen uses a <e2>disassembler</e2> to look at the raw assembly code.
# # sent = "The author 's of a keygen uses a disassembler to look at the raw assembly code ."
# # print(len(sent.split()))
#
# sent = "The imams were removed from a US Airways flight awaiting departure from the Minneapolis St. Paul airport."
# # sent = "I ate breakfast before watching TV."
# sent= "The horizontal branch is a part of the Hertzsprung -Russell (H-R) diagram that represents stars that burn helium in thier cores."
#
# sent = "In tele-operation, a human operator manipulates a control stick to generate a command signal so that a robot performs a specified task."
#
# sent = "the picture of madonna and child in glory with st sebastian and st rocco on the far wall of the church of campello sul clitunno was completed by one of his pupils from a preparatory drawing by lo spagna ."
#
# sent = "The car left the plant."
#
# sent = "This process passes on a health gene to the next generation."
#
# tokens = parser.tokenize(sent)
# print(tokens)
# #
# # # sdp = parser.get_sdp(dep_graph, 1, 2)
# # # print(sdp)
# #
# sdp = parser.get_SDP(' '.join(tokens), 6, 10)
# print(sdp)
#
# parser.visualize(' '.join(tokens))
#
