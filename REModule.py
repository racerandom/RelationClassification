# coding=utf-8

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import clear_output
import numpy as np
import random
import pdb
import gensim

from allennlp.modules.elmo import Elmo, batch_to_ids

import REData

import logging
logger = logging.getLogger('REOptimize')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_entity_hidden(hidden_states, entity_index, cat_method='sum'):
    e_h = [h[i[(i >= 0 ).nonzero().squeeze(dim=1)]] for h, i in zip(hidden_states, entity_index)]
    e_h_c = [catOverTime(h, cat_method, dim=0) for h in e_h]
    e_h_t = torch.stack(e_h_c)
    return e_h_t

def catOverTime(net_out, cat_method, dim=1):
    if not cat_method:
        net_out = net_out[:, -1, :]
    elif cat_method == 'max':
        net_out = net_out.max(dim=dim)[0]
    elif cat_method == 'mean':
        net_out = net_out.mean(dim=dim)
    elif cat_method == 'sum':
        net_out = net_out.sum(dim=dim)
    else:
        raise Exception("[Error] Unknown cat method")
    return net_out


class baseNN(nn.Module):

    def __init__(self, word_size, e1pos_size, e2pos_size, targ_size,
                 max_sent_len, pre_embed, **params):
        super(baseNN, self).__init__()

        self.params = params
        self.rnn_hidden_dim = self.params['rnn_hidden_dim']
        self.word_size = word_size
        self.e1pos_size = e1pos_size
        self.e2pos_size = e2pos_size
        self.targ_size = targ_size
        self.max_sent_len = max_sent_len

        if isinstance(pre_embed, np.ndarray):
            self.word_dim = pre_embed.shape[1]
            self.word_embeddings = REData.pre2embed(pre_embed, freeze_mode=self.params['freeze_mode'])


        self.e1pos_dim = self.params['pos_dim']
        if self.params['pos_dim']:
            self.e1pos_embeddings = nn.Embedding(e1pos_size, self.e1pos_dim, padding_idx=0)

        self.e2pos_dim = self.params['pos_dim']
        if self.params['pos_dim']:
            self.e2pos_embeddings = nn.Embedding(e2pos_size, self.e2pos_dim, padding_idx=0)

        self.input_dropout = nn.Dropout(p=self.params['input_dropout'])

    def init_rnn_hidden(self, batch_size, hidden_dim, num_layer=1, bidirectional=True):
        bi_num = 2 if bidirectional else 1
        return (torch.zeros(num_layer * bi_num, batch_size, hidden_dim // bi_num).to(device),
                torch.zeros(num_layer * bi_num, batch_size, hidden_dim // bi_num).to(device))


class baseRNN(baseNN):

    def __init__(self, word_size, e1pos_size, e2pos_size, targ_size,
                 max_sent_len, pre_embed, **params):

        super(baseRNN, self).__init__(word_size, e1pos_size, e2pos_size, targ_size,
                                    max_sent_len, pre_embed, **params)

        self.rnn_input_dim = self.word_dim + self.e1pos_dim + self.e2pos_dim

        self.rnn = nn.LSTM(self.rnn_input_dim,
                           self.rnn_hidden_dim // 2,
                           num_layers=self.params['rnn_layer'],
                           batch_first=True,
                           bidirectional=True)

        self.rnn_dropout = nn.Dropout(p=self.params['rnn_dropout'])

        self.fc1 = nn.Linear(self.rnn_hidden_dim, self.params['fc1_hidden_dim'])
        self.fc1_drop = nn.Dropout(p=self.params['fc1_dropout'])
        self.fc2 = nn.Linear(self.params['fc1_hidden_dim'], targ_size)


    def forward(self, *tensor_feats):

        batch_size = tensor_feats[0].shape[0]

        word_embed_input = self.word_embeddings(tensor_feats[0])
        e1pos_embed_input = self.e1pos_embeddings(tensor_feats[1])
        e2pos_embed_input = self.e2pos_embeddings(tensor_feats[2])

        rnn_input = torch.cat((word_embed_input, e1pos_embed_input, e2pos_embed_input), dim=2)

        rnn_hidden = self.init_rnn_hidden(batch_size,
                                          self.rnn_hidden_dim,
                                          num_layer=self.params['rnn_layer'])

        rnn_out, rnn_hidden = self.rnn(rnn_input, rnn_hidden)

        fc1_in = torch.cat((rnn_hidden[0][0, :, :], rnn_hidden[0][1, :, :]), dim=1)

        fc1_in = self.rnn_dropout(fc1_in)

        fc1_out = F.relu(self.fc1(fc1_in))
        fc1_out = self.fc1_drop(fc1_out)
        fc2_out = F.log_softmax(self.fc2(fc1_out), dim=1)

        return fc2_out


class attnDotRNN(baseNN):

    def __init__(self, word_size, e1pos_size, e2pos_size, targ_size,
                 max_sent_len, pre_embed, **params):

        super(attnDotRNN, self).__init__(word_size, e1pos_size, e2pos_size, targ_size,
                                    max_sent_len, pre_embed, **params)

        self.rnn_input_dim = self.word_dim + self.e1pos_dim + self.e2pos_dim

        self.rnn = nn.LSTM(self.rnn_input_dim,
                           self.rnn_hidden_dim // 2,
                           num_layers=self.params['rnn_layer'],
                           batch_first=True,
                           bidirectional=True)

        self.rnn_dropout = nn.Dropout(p=self.params['rnn_dropout'])

        self.fc1 = nn.Linear(self.rnn_hidden_dim, self.params['fc1_hidden_dim'])
        self.fc1_drop = nn.Dropout(p=self.params['fc1_dropout'])
        self.fc2 = nn.Linear(self.params['fc1_hidden_dim'], targ_size)

    def forward(self, word_t, e1pos_t, e2pos_t):

        batch_size = word_t.shape[0]

        word_embed_input = self.word_embeddings(word_t)
        e1pos_embed_input = self.e1pos_embeddings(e1pos_t)
        e2pos_embed_input = self.e2pos_embeddings(e2pos_t)

        rnn_input = torch.cat((word_embed_input, e1pos_embed_input, e2pos_embed_input), dim=2)

        rnn_hidden = self.init_rnn_hidden(batch_size,
                                          self.rnn_hidden_dim,
                                          num_layer=self.params['rnn_layer'])

        rnn_out, rnn_hidden = self.rnn(self.input_dropout(rnn_input), rnn_hidden)

        rnn_last = torch.cat((rnn_hidden[0][0, :, :], rnn_hidden[0][1, :, :]), dim=1)
        dot_prob = F.softmax(torch.bmm(rnn_out, rnn_last.unsqueeze(2)).squeeze(), dim=1)
        dot_out = torch.bmm(dot_prob.unsqueeze(1), rnn_out).squeeze()

        fc1_in = self.rnn_dropout(dot_out)

        fc1_out = F.relu(self.fc1(fc1_in))
        fc1_out = self.fc1_drop(fc1_out)
        fc2_out = F.log_softmax(self.fc2(fc1_out), dim=1)

        return fc2_out


class attnMatRNN(baseNN):

    def __init__(self, word_size, e1pos_size, e2pos_size, targ_size,
                 max_sent_len, pre_embed, **params):

        super(attnMatRNN, self).__init__(word_size, e1pos_size, e2pos_size, targ_size,
                                    max_sent_len, pre_embed, **params)

        self.rnn_input_dim = self.word_dim + self.e1pos_dim + self.e2pos_dim

        self.rnn = nn.LSTM(self.rnn_input_dim,
                           self.rnn_hidden_dim // 2,
                           num_layers=self.params['rnn_layer'],
                           batch_first=True,
                           bidirectional=True)

        self.attn_M = torch.nn.Parameter(torch.randn(self.rnn_hidden_dim, self.rnn_hidden_dim, requires_grad=True))

        self.rnn_dropout = nn.Dropout(p=self.params['rnn_dropout'])

        self.fc1 = nn.Linear(self.rnn_hidden_dim, self.params['fc1_hidden_dim'])
        self.fc1_drop = nn.Dropout(p=self.params['fc1_dropout'])
        self.fc2 = nn.Linear(self.params['fc1_hidden_dim'], targ_size)


    def forward(self, word_t, e1pos_t, e2pos_t):

        batch_size = word_t.shape[0]

        word_embed_input = self.word_embeddings(word_t)
        e1pos_embed_input = self.e1pos_embeddings(e1pos_t)
        e2pos_embed_input = self.e2pos_embeddings(e2pos_t)

        rnn_input = torch.cat((word_embed_input, e1pos_embed_input, e2pos_embed_input), dim=2)

        rnn_hidden = self.init_rnn_hidden(batch_size,
                                          self.rnn_hidden_dim,
                                          num_layer=self.params['rnn_layer'])

        rnn_out, rnn_hidden = self.rnn(self.input_dropout(rnn_input), rnn_hidden)

        rnn_last = torch.cat((rnn_hidden[0][0, :, :], rnn_hidden[0][1, :, :]), dim=1)
        attn_bM = self.attn_M.repeat(batch_size, 1, 1)
        attn_prob = F.softmax(torch.bmm(torch.bmm(rnn_out, attn_bM), rnn_last.unsqueeze(2)).squeeze(), dim=1)
        attn_out = torch.bmm(attn_prob.unsqueeze(1), rnn_out).squeeze()

        fc1_in = self.rnn_dropout(attn_out)

        fc1_out = F.relu(self.fc1(fc1_in))
        fc1_out = self.fc1_drop(fc1_out)
        fc2_out = F.log_softmax(self.fc2(fc1_out), dim=1)

        return fc2_out


class attnRNN(baseNN):

    def __init__(self, word_size, e1pos_size, e2pos_size, targ_size,
                 max_sent_len, pre_embed, **params):

        super(attnRNN, self).__init__(word_size, e1pos_size, e2pos_size, targ_size,
                                    max_sent_len, pre_embed, **params)

        self.rnn_input_dim = self.word_dim + self.e1pos_dim + self.e2pos_dim

        self.rnn = nn.LSTM(self.rnn_input_dim,
                           self.rnn_hidden_dim // 2,
                           num_layers=self.params['rnn_layer'],
                           batch_first=True,
                           bidirectional=True)

        self.attn_W = torch.nn.Parameter(torch.randn(self.rnn_hidden_dim // 2, requires_grad=True))

        self.rnn_dropout = nn.Dropout(p=self.params['rnn_dropout'])

        self.fc1 = nn.Linear(self.rnn_hidden_dim // 2, self.params['fc1_hidden_dim'])
        self.fc1_drop = nn.Dropout(p=self.params['fc1_dropout'])
        self.fc2 = nn.Linear(self.params['fc1_hidden_dim'], targ_size)


    def forward(self, word_t, e1pos_t, e2pos_t):

        batch_size = word_t.shape[0]

        word_embed_input = self.word_embeddings(word_t)
        e1pos_embed_input = self.e1pos_embeddings(e1pos_t)
        e2pos_embed_input = self.e2pos_embeddings(e2pos_t)

        rnn_input = torch.cat((word_embed_input, e1pos_embed_input, e2pos_embed_input), dim=2)

        rnn_hidden = self.init_rnn_hidden(batch_size,
                                          self.rnn_hidden_dim,
                                          num_layer=self.params['rnn_layer'])

        rnn_out, _ = self.rnn(self.input_dropout(rnn_input), rnn_hidden)

        rnn_out = rnn_out[:,:,:self.rnn_hidden_dim // 2] + rnn_out[:,:,self.rnn_hidden_dim // 2:]

        attn_bW = self.attn_W.repeat(batch_size, 1)
        attn_alpha = torch.bmm(rnn_out, attn_bW.unsqueeze(2))
        attn_prob = F.softmax(attn_alpha.squeeze(), dim=1)
        attn_out = F.tanh(torch.bmm(attn_prob.unsqueeze(1), rnn_out))

        fc1_in = self.rnn_dropout(attn_out.squeeze())

        fc1_out = F.relu(self.fc1(fc1_in))
        fc1_out = self.fc1_drop(fc1_out)
        fc2_out = F.log_softmax(self.fc2(fc1_out), dim=1)

        return fc2_out


class entiAttnDotRNN(baseNN):

    def __init__(self, word_size, e1pos_size, e2pos_size, targ_size,
                 max_sent_len, pre_embed, **params):

        super(entiAttnDotRNN, self).__init__(word_size, e1pos_size, e2pos_size, targ_size,
                                    max_sent_len, pre_embed, **params)

        self.rnn_input_dim = self.word_dim + self.e1pos_dim + self.e2pos_dim

        self.rnn = nn.LSTM(self.rnn_input_dim,
                           self.rnn_hidden_dim // 2,
                           num_layers=self.params['rnn_layer'],
                           batch_first=True,
                           bidirectional=True)

        self.rnn_dropout = nn.Dropout(p=self.params['rnn_dropout'])

        self.fc1 = nn.Linear(self.rnn_hidden_dim * 2, self.params['fc1_hidden_dim'])
        self.fc1_drop = nn.Dropout(p=self.params['fc1_dropout'])
        self.fc2 = nn.Linear(self.params['fc1_hidden_dim'], targ_size)

    def forward(self, *tensor_feats):

        batch_size = tensor_feats[0].shape[0]

        word_embed_input = self.word_embeddings(tensor_feats[0])
        e1pos_embed_input = self.e1pos_embeddings(tensor_feats[1])
        e2pos_embed_input = self.e2pos_embeddings(tensor_feats[2])

        rnn_input = torch.cat((word_embed_input, e1pos_embed_input, e2pos_embed_input), dim=2)

        rnn_hidden = self.init_rnn_hidden(batch_size,
                                          self.rnn_hidden_dim,
                                          num_layer=self.params['rnn_layer'])

        rnn_out, rnn_hidden = self.rnn(self.input_dropout(rnn_input), rnn_hidden)

        e1_hidden = batch_entity_hidden(rnn_out, tensor_feats[3])
        e2_hidden = batch_entity_hidden(rnn_out, tensor_feats[4])

        e1_dot_prob = F.softmax(torch.bmm(rnn_out, e1_hidden.unsqueeze(2)).squeeze(), dim=1)
        e1_dot_out = torch.bmm(e1_dot_prob.unsqueeze(1), rnn_out).squeeze()

        e2_dot_prob = F.softmax(torch.bmm(rnn_out, e2_hidden.unsqueeze(2)).squeeze(), dim=1)
        e2_dot_out = torch.bmm(e2_dot_prob.unsqueeze(1), rnn_out).squeeze()

        fc1_in = self.rnn_dropout(torch.cat((e1_dot_out, e2_dot_out), dim=1))

        fc1_out = F.relu(self.fc1(fc1_in))
        fc1_out = self.fc1_drop(fc1_out)
        fc2_out = F.log_softmax(self.fc2(fc1_out), dim=1)

        return fc2_out