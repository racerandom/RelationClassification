# coding=utf-8
import xml.etree.ElementTree as ET
from RESyntaxer import RESyntaxer

corenlp = RESyntaxer()

class Relation():

    def __init__(self, id=None, sent=None, comm=None, rel=None):
        self.id = id
        self.sent = sent
        self.rel = rel
        self.comm = comm
        self.tokens = []
        self.e1_tids = []
        self.e2_tids = []
        self.feat_inputs = {}

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, id):
        self.__id = id

    @property
    def sent(self):
        return self.__sent

    @sent.setter
    def sent(self, sent):
        self.__sent = sent

    @property
    def rel(self):
        return self.__rel

    @rel.setter
    def rel(self, rel):
        self.__rel = rel

    @property
    def comm(self):
        return self.__comm

    @comm.setter
    def comm(self, comm):
        self.__comm = comm

    @property
    def e1_sur(self):
        return ' '.join([self.tokens[tid] for tid in self.e1_tids])

    @property
    def e2_sur(self):
        return ' '.join([self.tokens[tid] for tid in self.e2_tids])

    def tokenize_sent(self, PI=False):
        word_tokenize = corenlp.get_token

        text = "<S>%s</S>" % self.sent
        text = text.encode('utf-8')
        text = text.replace(b'&', b'&amp;')
        root = ET.fromstring(text)
        tok_id = 0

        if root.text:
            curr_toks = word_tokenize(root.text)
            self.tokens.extend(curr_toks)
            tok_id += len(curr_toks)

        for index, entity in enumerate(root):
            if entity.text:
                if PI:
                    self.tokens.append('<%s>' % entity.tag)
                    tok_id += 1
                for tok in word_tokenize(entity.text):
                    self.tokens.append(tok)
                    entity_ids = getattr(self, 'e%i_tids' % (index + 1))
                    entity_ids.append(tok_id)
                    tok_id += 1
                if PI:
                    self.tokens.append('</%s>' % entity.tag)
                    tok_id += 1
            if entity.tail:
                curr_toks = word_tokenize(entity.tail)
                self.tokens.extend(curr_toks)
                tok_id += len(curr_toks)

    def is_entity_feats(self):
        feats = []
        for index, token in enumerate(self.tokens):
            feats.append([0 if index not in self.e1_tids else 1,
                          0 if index not in self.e2_tids else 1])
        return feats

    def attach_feats(self, feat_name, feats):
        self.feat_inputs[feat_name] = feats


    def print_samp(self):
        print("A relation sample with id=%s\n"
              "sent=\"%s\"\n"
              "rel=%s\n" % (self.id, self.sent, self.rel))


if __name__ == '__main__':
    sent = "Narrative <e1>identity</e1> takes part in the story's <e2>movement</e2>, " \
           "in the dialectic between order and disorder."
    rel_samp = Relation()
    rel_samp.sent = sent
    rel_samp.tokenize_sent(PI=True)
    print(rel_samp.tokens)
    print(rel_samp.e1_tids)
    print(rel_samp.e1_sur)
    print(rel_samp.e2_tids)
    print(rel_samp.e2_sur)