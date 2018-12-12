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


def batch_entity_hidden(hidden_states, entity_index, cat_entity='sum'):
    e_h = [h[i[(i >= 0 ).nonzero().squeeze(dim=1)]] for h, i in zip(hidden_states, entity_index)]
    e_h_c = [catOverTime(h, cat_entity, dim=0) for h in e_h]
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

    def __init__(self, word_size, targ_size,
                 max_sent_len, pre_embed, **params):
        super(baseNN, self).__init__()

        self.params = params
        self.rnn_hidden_dim = self.params['rnn_hidden_dim']
        self.word_size = word_size
        self.targ_size = targ_size
        self.max_sent_len = max_sent_len

        if isinstance(pre_embed, np.ndarray):
            self.word_dim = pre_embed.shape[1]
            self.word_embeddings = REData.pre2embed(pre_embed, freeze_mode=self.params['freeze_mode'])

        self.input_dropout = nn.Dropout(p=self.params['input_dropout'])

    def init_rnn_hidden(self, batch_size, hidden_dim, num_layer=1, bidirectional=True):
        bi_num = 2 if bidirectional else 1
        return (torch.zeros(num_layer * bi_num, batch_size, hidden_dim // bi_num).to(device),
                torch.zeros(num_layer * bi_num, batch_size, hidden_dim // bi_num).to(device))

    def attn_dot_input(self, *tensor_feats):

        batch_size = tensor_feats[0].shape[0]

        word_embed_M = self.word_embeddings(tensor_feats[0]) # (batch, max_sent, word_dim)

        e1_embed = batch_entity_hidden(word_embed_M, tensor_feats[1]) # (batch, word_dim)
        e2_embed = batch_entity_hidden(word_embed_M, tensor_feats[2]) # (batch, word_dim)

        e1_dot_weight = F.softmax(torch.bmm(word_embed_M, e1_embed.unsqueeze(2)).squeeze(), dim=1)
        e2_dot_weight = F.softmax(torch.bmm(word_embed_M, e2_embed.unsqueeze(2)).squeeze(), dim=1)
        dot_weight = (e1_dot_weight + e2_dot_weight) /2 # average weights
        attn_applied_M = torch.bmm(dot_weight.unsqueeze(1), word_embed_M).squeeze()

        return attn_applied_M


class attnLayer(nn.Module):

    def __init__(self, max_sent_len, attn_input_dim, **params):
        super(attnLayer, self).__init__()
        self.attn_fc1 = nn.Linear(attn_input_dim, params['attn_hidden_dim'])
        self.hidden2prob = nn.Linear(params['attn_hidden_dim'], max_sent_len)

    def forward(self, *tensor_input):
        attn_fc1_out = F.relu(self.attn_fc1(tensor_input))
        prob_out = F.softmax(self.hidden2prob(attn_fc1_out))
        return prob_out

class dotAttn(nn.Module):

    def __init__(self):
        super(dotAttn, self).__init__()

    def forward(self, *tensor_feats):

        input_embed_M = tensor_feats[0]

        e1_embed = batch_entity_hidden(input_embed_M, tensor_feats[1])  # (batch, word_dim)
        e2_embed = batch_entity_hidden(input_embed_M, tensor_feats[2])  # (batch, word_dim)

        e1_dot_weight = F.softmax(torch.bmm(input_embed_M, e1_embed.unsqueeze(2)).squeeze(), dim=1)
        e2_dot_weight = F.softmax(torch.bmm(input_embed_M, e2_embed.unsqueeze(2)).squeeze(), dim=1)

        dot_weight = (e1_dot_weight + e2_dot_weight) / 2  # average weights
        attn_applied_M = torch.bmm(dot_weight.unsqueeze(1), input_embed_M).squeeze()

        return attn_applied_M


class matAttn(nn.Module):

    def __init__(self, hidden_dim):
        super(matAttn, self).__init__()
        self.attn_M = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim, requires_grad=True))

    def forward(self, *tensor_feats):

        batch_size = tensor_feats[0].shape[0]

        input_embed_M = tensor_feats[0]

        e1_embed = batch_entity_hidden(input_embed_M, tensor_feats[1])  # (batch, word_dim)
        e2_embed = batch_entity_hidden(input_embed_M, tensor_feats[2])  # (batch, word_dim)

        attn_bM = self.attn_M.repeat(batch_size, 1, 1)

        input_weight_M = torch.bmm(input_embed_M, attn_bM)

        e1_mat_weight = F.softmax(torch.bmm(input_weight_M, e1_embed.unsqueeze(2)).squeeze(), dim=1)
        e2_mat_weight = F.softmax(torch.bmm(input_weight_M, e2_embed.unsqueeze(2)).squeeze(), dim=1)

        mat_weight = (e1_mat_weight + e2_mat_weight) / 2  # average weights

        attn_applied_M = torch.bmm(mat_weight.unsqueeze(1), input_embed_M).squeeze()

        return attn_applied_M



class baseRNN(baseNN):

    def __init__(self, word_size, targ_size,
                 max_sent_len, pre_embed, **params):

        super(baseRNN, self).__init__(word_size, targ_size,
                                      max_sent_len, pre_embed, **params)

        self.rnn_input_dim = self.word_dim

        self.rnn = nn.LSTM(self.rnn_input_dim,
                           self.rnn_hidden_dim // 2,
                           num_layers=self.params['rnn_layer'],
                           batch_first=True,
                           bidirectional=True)

        self.rnn_dropout = nn.Dropout(p=self.params['rnn_dropout'])

        self.fc = nn.Linear(self.rnn_hidden_dim, targ_size)

    def forward(self, *tensor_feats):

        batch_size = tensor_feats[0].shape[0]

        word_embed_input = self.word_embeddings(tensor_feats[0])

        rnn_input = self.input_dropout(word_embed_input)

        rnn_hidden = self.init_rnn_hidden(batch_size,
                                          self.rnn_hidden_dim,
                                          num_layer=self.params['rnn_layer'])

        rnn_out, rnn_hidden = self.rnn(rnn_input, rnn_hidden)

        # fc_in = torch.cat(torch.unbind(rnn_hidden[0],dim=0), dim=1) ## last hidden state

        fc_in = catOverTime(rnn_out, 'max')

        softmax_out = F.log_softmax(self.fc(self.rnn_dropout(fc_in)), dim=1)

        return softmax_out


class attnDotRNN(baseNN):

    def __init__(self, word_size, targ_size,
                 max_sent_len, pre_embed, **params):

        super(attnDotRNN, self).__init__(word_size, targ_size,
                                         max_sent_len, pre_embed, **params)

        self.rnn_input_dim = self.word_dim

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

        rnn_input = self.input_dropout(word_embed_input)

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

    def __init__(self, word_size, targ_size,
                 max_sent_len, pre_embed, **params):

        super(attnMatRNN, self).__init__(word_size, targ_size,
                                         max_sent_len, pre_embed, **params)

        self.rnn_input_dim = self.word_dim

        self.rnn = nn.LSTM(self.rnn_input_dim,
                           self.rnn_hidden_dim // 2,
                           num_layers=self.params['rnn_layer'],
                           batch_first=True,
                           bidirectional=True)

        self.attn_M = torch.nn.Parameter(torch.randn(self.rnn_hidden_dim, self.rnn_hidden_dim, requires_grad=True))

        self.rnn_dropout = nn.Dropout(p=self.params['rnn_dropout'])

        self.fc_dropout = nn.Dropout(p=self.params['fc1_dropout'])
        self.fc = nn.Linear(self.rnn_hidden_dim, targ_size)


    def forward(self, *tensor_feats):

        batch_size = tensor_feats[0].shape[0]

        word_embed_input = self.word_embeddings(tensor_feats[0])

        rnn_input = self.input_dropout(word_embed_input)

        rnn_hidden = self.init_rnn_hidden(batch_size,
                                          self.rnn_hidden_dim,
                                          num_layer=self.params['rnn_layer'])

        rnn_out, rnn_hidden = self.rnn(self.input_dropout(rnn_input), rnn_hidden)

        rnn_out = self.rnn_dropout(rnn_out)

        rnn_last = torch.cat((rnn_hidden[0][0, :, :], rnn_hidden[0][1, :, :]), dim=1)
        attn_bM = self.attn_M.repeat(batch_size, 1, 1)
        attn_prob = F.softmax(torch.bmm(torch.bmm(rnn_out, attn_bM), rnn_last.unsqueeze(2)).squeeze(), dim=1)
        attn_out = torch.bmm(attn_prob.unsqueeze(1), rnn_out).squeeze()

        attn_out = self.fc_dropout(attn_out.squeeze())

        fc_out = F.log_softmax(self.fc(attn_out), dim=1)

        return fc_out


class attnRNN(baseNN):

    def __init__(self, word_size, targ_size,
                 max_sent_len, pre_embed, **params):

        super(attnRNN, self).__init__(word_size, targ_size,
                                    max_sent_len, pre_embed, **params)

        self.rnn_input_dim = self.word_dim

        self.rnn = nn.LSTM(self.rnn_input_dim,
                           self.rnn_hidden_dim // 2,
                           num_layers=self.params['rnn_layer'],
                           batch_first=True,
                           bidirectional=True)

        self.attn_W = torch.nn.Parameter(torch.randn(self.rnn_hidden_dim // 2, requires_grad=True))

        self.rnn_dropout = nn.Dropout(p=self.params['rnn_dropout'])

        self.fc_drop = nn.Dropout(p=self.params['fc1_dropout'])
        self.fc = nn.Linear(self.rnn_hidden_dim // 2, targ_size)


    def forward(self, *tensor_feats):

        batch_size = tensor_feats[0].shape[0]

        word_embed_input = self.word_embeddings(tensor_feats[0])

        rnn_input = self.input_dropout(word_embed_input)

        rnn_hidden = self.init_rnn_hidden(batch_size,
                                          self.rnn_hidden_dim,
                                          num_layer=self.params['rnn_layer'])

        rnn_out, _ = self.rnn(self.input_dropout(rnn_input), rnn_hidden)

        rnn_out = rnn_out[:,:,:self.rnn_hidden_dim // 2] + rnn_out[:,:,self.rnn_hidden_dim // 2:]

        rnn_out = self.rnn_dropout(rnn_out)

        attn_bW = self.attn_W.repeat(batch_size, 1)
        attn_alpha = torch.bmm(rnn_out, attn_bW.unsqueeze(2))
        attn_prob = F.softmax(attn_alpha.squeeze(), dim=1)
        attn_out = F.tanh(torch.bmm(attn_prob.unsqueeze(1), rnn_out))

        attn_out = self.fc_drop(attn_out.squeeze())

        fc_out = F.log_softmax(self.fc(attn_out), dim=1)

        return fc_out


class entiAttnDotRNN(baseNN):

    def __init__(self, word_size, targ_size,
                 max_sent_len, pre_embed, **params):

        super(entiAttnDotRNN, self).__init__(word_size, targ_size,
                                    max_sent_len, pre_embed, **params)

        self.rnn_input_dim = self.word_dim

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

        rnn_input = self.input_dropout(word_embed_input)

        rnn_hidden = self.init_rnn_hidden(batch_size,
                                          self.rnn_hidden_dim,
                                          num_layer=self.params['rnn_layer'])

        rnn_out, rnn_hidden = self.rnn(self.input_dropout(rnn_input), rnn_hidden)

        rnn_out = self.rnn_dropout(rnn_out)

        e1_hidden = batch_entity_hidden(rnn_out, tensor_feats[1])
        e2_hidden = batch_entity_hidden(rnn_out, tensor_feats[2])

        e1_dot_prob = F.softmax(torch.bmm(rnn_out, e1_hidden.unsqueeze(2)).squeeze(), dim=1)
        e1_dot_out = torch.bmm(e1_dot_prob.unsqueeze(1), rnn_out).squeeze()

        e2_dot_prob = F.softmax(torch.bmm(rnn_out, e2_hidden.unsqueeze(2)).squeeze(), dim=1)
        e2_dot_out = torch.bmm(e2_dot_prob.unsqueeze(1), rnn_out).squeeze()

        fc1_in = torch.cat((e1_dot_out, e2_dot_out), dim=1)

        fc1_out = F.relu(self.fc1(fc1_in))
        fc1_out = self.fc1_drop(fc1_out)
        fc2_out = F.log_softmax(self.fc2(fc1_out), dim=1)

        return fc2_out


class entiAttnMatRNN(baseNN):

    def __init__(self, word_size, targ_size,
                 max_sent_len, pre_embed, **params):

        super(entiAttnMatRNN, self).__init__(word_size, targ_size,
                                    max_sent_len, pre_embed, **params)

        self.rnn_input_dim = self.word_dim

        self.rnn = nn.LSTM(self.rnn_input_dim,
                           self.rnn_hidden_dim // 2,
                           num_layers=self.params['rnn_layer'],
                           batch_first=True,
                           bidirectional=True)

        self.rnn_dropout = nn.Dropout(p=self.params['rnn_dropout'])

        self.attn_M = torch.nn.Parameter(torch.randn(self.rnn_hidden_dim, self.rnn_hidden_dim, requires_grad=True))

        self.fc1 = nn.Linear(self.rnn_hidden_dim * 2, self.params['fc1_hidden_dim'])
        self.fc1_drop = nn.Dropout(p=self.params['fc1_dropout'])
        self.fc2 = nn.Linear(self.params['fc1_hidden_dim'], targ_size)

    def forward(self, *tensor_feats):

        batch_size = tensor_feats[0].shape[0]

        word_embed_input = self.word_embeddings(tensor_feats[0])

        rnn_input = self.input_dropout(word_embed_input)

        rnn_hidden = self.init_rnn_hidden(batch_size,
                                          self.rnn_hidden_dim,
                                          num_layer=self.params['rnn_layer'])

        rnn_out, rnn_hidden = self.rnn(self.input_dropout(rnn_input), rnn_hidden)

        rnn_out = self.rnn_dropout(rnn_out)

        attn_bM = self.attn_M.repeat(batch_size, 1, 1)

        e1_hidden = batch_entity_hidden(rnn_out, tensor_feats[1])
        e2_hidden = batch_entity_hidden(rnn_out, tensor_feats[2])

        rnn_out_mat = torch.bmm(rnn_out, attn_bM)

        e1_mat_prob = F.softmax(torch.bmm(rnn_out_mat, e1_hidden.unsqueeze(2)).squeeze(), dim=1)
        e1_mat_out = torch.bmm(e1_mat_prob.unsqueeze(1), rnn_out).squeeze()

        e2_mat_prob = F.softmax(torch.bmm(rnn_out_mat, e2_hidden.unsqueeze(2)).squeeze(), dim=1)
        e2_mat_out = torch.bmm(e2_mat_prob.unsqueeze(1), rnn_out).squeeze()

        fc1_in = torch.cat((e1_mat_out, e2_mat_out), dim=1)

        fc1_out = F.relu(self.fc1(fc1_in))
        fc1_out = self.fc1_drop(fc1_out)
        fc2_out = F.log_softmax(self.fc2(fc1_out), dim=1)

        return fc2_out


class mulEntiAttnDotRNN(baseNN):

    def __init__(self, word_size, targ_size,
                 max_sent_len, pre_embed, **params):

        super(mulEntiAttnDotRNN, self).__init__(word_size, targ_size,
                                                max_sent_len, pre_embed, **params)

        self.rnn_input_dim = self.word_dim

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

        rnn_input = self.input_dropout(word_embed_input)

        rnn_hidden = self.init_rnn_hidden(batch_size,
                                          self.rnn_hidden_dim,
                                          num_layer=self.params['rnn_layer'])

        rnn_out, rnn_hidden = self.rnn(self.input_dropout(rnn_input), rnn_hidden)

        rnn_out = self.rnn_dropout(rnn_out)

        e1_hidden = batch_entity_hidden(rnn_out, tensor_feats[1])
        e2_hidden = batch_entity_hidden(rnn_out, tensor_feats[2])

        attn_dot_scores = torch.bmm(rnn_out, torch.add(e1_hidden, e2_hidden).unsqueeze(2)).squeeze()
        dot_prob = F.softmax(attn_dot_scores, dim=1)
        dot_out = torch.bmm(dot_prob.unsqueeze(1), rnn_out).squeeze()

        fc1_out = F.relu(self.fc1(dot_out))
        fc1_out = self.fc1_drop(fc1_out)
        fc2_out = F.log_softmax(self.fc2(fc1_out), dim=1)

        return fc2_out