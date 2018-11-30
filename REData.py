# coding=utf-8

import os
import numpy as np
import gensim
import time
import pickle
import torch
import torch.nn as nn

## inner library
from REObject import Relation

def data_reader(filename):
    with open(filename, 'r') as fi:

        rel_list = []
        rel_samp = Relation()
        line_flag = 0

        for line in fi:

            if not line.strip() and line_flag == 3:
                rel_samp = Relation()
                line_flag = 0
                continue

            if line_flag == 0:
                toks = line.strip().split('\t')
                rel_samp.id = toks[0]
                rel_samp.sent = toks[1].strip('\"')
            elif line_flag == 1:
                rel_samp.rel = line.strip()
            elif line_flag == 2:
                rel_samp.comm = line.strip("Comment:")
                rel_list.append(rel_samp)
            else:
                raise Exception("[ERROR]Unknown line_flag to operate...[%i]" % line_flag)

            line_flag += 1
    return rel_list


def prepare_feats(rel_data):
    for rel in rel_data:
        rel.tokenize_sent()
        rel.attach_feats('word_sent', rel.tokens)
        pos_feat = rel.is_entity_feats()
        rel.attach_feats('pos_sent', pos_feat)


def load_pre_embed(embed_file, binary):
    if embed_file and os.path.isfile(os.path.join(os.getenv("HOME"), embed_file)):
        start_time = time.time()
        pre_embed = gensim.models.KeyedVectors.load_word2vec_format(embed_file, binary=binary)
        pre_word2ix = {}
        for word, value in pre_embed.vocab.items():
            pre_word2ix[word] = value.index
        print("[Embedding] Successfully load the pre-trained embedding file '%s' in %i seconds..." % (embed_file,
                                                                                                      time.time() - start_time))
        return pre_word2ix, pre_embed.vectors
    else:
        raise Exception("[ERROR]Cannot find the pre-trained embedding file...")


def pre_embed_to_weight(word2ix, embed_file, binary=True):
    pre_word2ix, pre_weights = load_pre_embed(embed_file, binary)
    count = 0
    word_dim = pre_weights.shape[1]
    weights = []
    for key, ix in sorted(word2ix.items(), key=lambda x: x[1]):
        if ix == 0:
            weights.append(np.zeros(word_dim))
        else:
            if key in pre_word2ix:
                count += 1
                weights.append(pre_weights[pre_word2ix[key]])
            elif key.lower() in pre_word2ix:
                count += 1
                weights.append(pre_weights[pre_word2ix[key.lower()]])
            else:
                weights.append(np.random.uniform(-1, 1, word_dim))
    slim_weights = np.stack(weights, axis=0)
    print("[Embedding] Successfully the slim embedding weights from %i to %i words, "
          "%i words (%.3f%%) are covered" % (len(pre_word2ix),
                                             len(word2ix),
                                             count,
                                             count/len(word2ix)))
    return slim_weights


def pre2embed(pre_vectors, freeze_mode):
    pre_weights = torch.FloatTensor(pre_vectors)
    return nn.Embedding.from_pretrained(pre_weights, freeze=freeze_mode)


def feat_to_ix(feats, feat2ix=None):
    if not feat2ix:
        feat2ix = {'zeropadding': 0}
    for feat_sample in feats:
        for tok in feat_sample:
            feat2ix.setdefault(tok, len(feat2ix))
    return feat2ix

def targ_to_ix(targs):
    targ2ix = {}
    for targ in targs:
        targ2ix.setdefault(targ, len(targ2ix))
    return targ2ix


def pickle_data(data, pickle_file='data/temp.pkl'):
    with open(pickle_file, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(data, f)
    print("Successfully save '%s'..." % pickle_file)


def load_pickle(pickle_file='data/temp.pkl'):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    print("Successfully load data from pickle file '%s'..." % pickle_file)
    return data

## convert a token to an index number
def tok2ix(tok, to_ix, unk_ix):
    return to_ix[tok] if tok in to_ix else unk_ix

## convert 1D token sequences to token_index sequences
def prepare_seq_1d(seq_1d, to_ix, unk_ix=0):
    ix_seq_1d = [tok2ix(tok, to_ix, unk_ix) for tok in seq_1d]
    return ix_seq_1d

## convert 2D token sequences to token_index sequences
def prepare_seq_2d(seq_2d, to_ix, unk_ix=0):
    ix_seq_2d = [[tok2ix(tok, to_ix, unk_ix) for tok in seq_1d] for seq_1d in seq_2d]
    return ix_seq_2d

## padding 2D index sequences to a fixed given length
def padding_2d(seq_2d, max_seq_len, padding=0, direct='right'):

    for seq_1d in seq_2d:
        for i in range(0, max_seq_len - len(seq_1d)):
            if direct in ['right']:
                seq_1d.append(padding)
            else:
                seq_1d.insert(0, padding)
    return seq_2d


def max_len_2d(seq_2d):
    return max([len(seq) for seq in seq_2d])


def save_all_data():

    train_file = "data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT"
    test_file = "data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"

    train_data = data_reader(train_file)
    prepare_feats(train_data)
    train_rels = train_data[:7109]
    val_rels = train_data[7109:]
    pickle_data(train_rels, pickle_file='data/train.pkl')
    pickle_data(val_rels, pickle_file='data/val.pkl')

    test_data = data_reader(test_file)
    prepare_feats(test_data)
    pickle_data(test_data, pickle_file='data/test.pkl')


def generate_feat2ix(train_file):

    train_data = load_pickle(pickle_file=train_file)
    word_feat = [rel.feat_inputs['word_sent'] for rel in train_data]
    e1pos_feat = [[str(pos[0]) for pos in rel.feat_inputs['pos_sent']] for rel in train_data]
    e2pos_feat = [[str(pos[1]) for pos in rel.feat_inputs['pos_sent']] for rel in train_data]
    max_sent_len = max_len_2d(word_feat)
    rel_label = [rel.rel for rel in train_data]

    word2ix = feat_to_ix(word_feat)
    e1pos2ix = feat_to_ix(e1pos_feat)
    e2pos2ix = feat_to_ix(e2pos_feat)
    targ2ix = targ_to_ix(rel_label)

    print("[data2ix] word vocab %i, e1pos vocab %i, e2pos vocab %i, "
          "targ size %i, max sent len %i\n" % (len(word2ix),
                                               len(e1pos2ix),
                                               len(e2pos2ix),
                                               len(targ2ix),
                                               max_sent_len))
    return word2ix, e1pos2ix, e2pos2ix, targ2ix, max_sent_len


def generate_data(data_file, word2ix, e1pos2ix, e2pos2ix, targ2ix, max_sent_len):

    rel_data = load_pickle(pickle_file=data_file)
    word_feat = [rel.feat_inputs['word_sent'] for rel in rel_data]
    word_list = prepare_seq_2d(word_feat, word2ix)
    word_tensor = torch.tensor(padding_2d(word_list, max_sent_len))

    e1pos_feat = [[str(pos[0]) for pos in rel.feat_inputs['pos_sent']] for rel in rel_data]
    e2pos_feat = [[str(pos[1]) for pos in rel.feat_inputs['pos_sent']] for rel in rel_data]

    e1pos_list = prepare_seq_2d(e1pos_feat, e1pos2ix)
    e1pos_tensor = torch.tensor(padding_2d(e1pos_list, max_sent_len))

    e2pos_list = prepare_seq_2d(e2pos_feat, e2pos2ix)
    e2pos_tensor = torch.tensor(padding_2d(e2pos_list, max_sent_len))

    targs = [rel.rel for rel in rel_data]
    targ_tensor = torch.tensor(prepare_seq_1d(targs, targ2ix))

    print("[Data] '%s' is generated with: word %s, e1pos %s, e2pos %s, targs %s\n" % (data_file,
                                                                                      word_tensor.shape,
                                                                                      e1pos_tensor.shape,
                                                                                      e2pos_tensor.shape,
                                                                                      targ_tensor.shape))


    return word_tensor, e1pos_tensor, e2pos_tensor, targ_tensor


if __name__ == '__main__':

    # save_all_data()

    train_file = "data/train.pkl"
    val_file = "data/val.pkl"
    test_file = "data/test.pkl"

    word2ix, e1pos2ix, e2pos2ix, targ2ix, max_sent_len = generate_feat2ix(train_file)

    train_word, train_e1pos, train_e2pos, train_targs = generate_data(train_file,
                                                                      word2ix,
                                                                      e1pos2ix,
                                                                      e2pos2ix,
                                                                      targ2ix,
                                                                      max_sent_len)

    val_word, val_e1pos, val_e2pos, val_targs = generate_data(val_file,
                                                              word2ix,
                                                              e1pos2ix,
                                                              e2pos2ix,
                                                              targ2ix,
                                                              max_sent_len)

    test_word, test_e1pos, test_e2pos, test_targs = generate_data(test_file,
                                                                  word2ix,
                                                                  e1pos2ix,
                                                                  e2pos2ix,
                                                                  targ2ix,
                                                                  max_sent_len)

    embed_file = "/Users/fei-c/Resources/embed/deps.words.bin"
    embed_weights = pre_embed_to_weight(word2ix, embed_file, binary=True)

    pickle_data(embed_weights, pickle_file='data/deps.words.embed')





