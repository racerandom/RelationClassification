# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from allennlp.modules.elmo import Elmo, batch_to_ids

import REData
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_entity_hidden(hidden_states, entity_index, cat_entity='sum'):
    e_h = [h[i] for h, i in zip(hidden_states, entity_index)]
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

        e1_dot_weight = F.softmax(torch.bmm(input_embed_M, e1_embed.unsqueeze(2)).squeeze(2), dim=1)
        e2_dot_weight = F.softmax(torch.bmm(input_embed_M, e2_embed.unsqueeze(2)).squeeze(2), dim=1)

        dot_weight = (e1_dot_weight + e2_dot_weight) / 2  # average weights

        return dot_weight


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

        input_attn_bM = torch.bmm(input_embed_M, attn_bM)

        e1_mat_weight = F.softmax(torch.bmm(input_attn_bM, e1_embed.unsqueeze(2)).squeeze(2), dim=1)
        e2_mat_weight = F.softmax(torch.bmm(input_attn_bM, e2_embed.unsqueeze(2)).squeeze(2), dim=1)

        attn_weight = (e1_mat_weight + e2_mat_weight) / 2  # average weights

        return attn_weight


class ranking_layer(nn.Module):

    def __init__(self, model_out_dim, targ_size):
        super(ranking_layer, self).__init__()
        self.targ_size = targ_size
        self.targ_dim = model_out_dim
        r = torch.sqrt(torch.tensor(6 / (targ_size + model_out_dim)))
        self.targ_weight = torch.nn.Parameter(torch.empty(model_out_dim, targ_size).uniform_(-r, r))

    def forward(self, batch_model_out):

        # batch_out: batch_size * out_dim, batch_gold: batch * 1

        batch_size = batch_model_out.shape[0]

        batch_targ_weights = self.targ_weight.repeat(batch_size, 1, 1)  # batch_size * out_dim * targ_size

        batch_pred_scores = torch.bmm(batch_model_out.unsqueeze(1), batch_targ_weights).squeeze(1)  # batch * targ_size

        return batch_pred_scores


def get_neg_scores(batch_out, batch_gold, batch_size, targ_size):
    # batch_out: batch_size * out_dim, batch_gold: batch * 1
    mask = torch.ones_like(batch_out, dtype=torch.uint8)  # batch_size * out_dim

    for i in range(batch_gold.shape[0]):
        mask[i][batch_gold[i]] = 0

    batch_neg_out = batch_out.masked_select(mask)

    assert batch_neg_out.shape[0] == batch_size * (targ_size - 1)

    batch_neg_scores = batch_neg_out.view(batch_size, targ_size - 1).max(dim=1)[0]

    return batch_neg_scores


def ranking_loss(batch_pred_scores, batch_gold, gamma=2, margin_pos=2.5, margin_neg=0.5):

    batch_size, targ_size = batch_pred_scores.shape

    batch_pos_scores = batch_pred_scores.gather(1, batch_gold.unsqueeze(1)).squeeze(1)  # batch * 1

    batch_neg_scores = get_neg_scores(batch_pred_scores, batch_gold, batch_size, targ_size)  # batch * 1

    loss = torch.log(1 + torch.exp(gamma * (margin_pos - batch_pos_scores))) + \
           torch.log(1 + torch.exp(gamma * (margin_neg + batch_neg_scores)))

    return loss.mean(dim=0)


class softmax_layer(nn.Module):

    def __init__(self, model_out_dim, targ_size):
        super(softmax_layer).__init__()
        self.fc = nn.Linear(model_out_dim, targ_size)

    def forward(self, tensor_input):
        fc_out = self.fc(tensor_input)
        softmax_out = F.log_softmax(fc_out)
        return softmax_out


class attnInBaseRNN(baseNN):

    def __init__(self, word_size, targ_size,
                 max_sent_len, pre_embed, **params):

        super(attnInBaseRNN, self).__init__(word_size, targ_size,
                                            max_sent_len, pre_embed, **params)

        self.rnn_input_dim = self.word_dim

        # self.attn_input = matAttn(self.rnn_input_dim)
        self.attn_input = dotAttn()

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

        attn_weight = self.attn_input(*(word_embed_input, tensor_feats[1], tensor_feats[2]))

        attn_applied_input = attn_weight.unsqueeze(2) * word_embed_input

        print(attn_applied_input.shape)

        rnn_input = self.input_dropout(attn_applied_input)

        rnn_hidden = self.init_rnn_hidden(batch_size,
                                          self.rnn_hidden_dim,
                                          num_layer=self.params['rnn_layer'])

        rnn_out, rnn_hidden = self.rnn(rnn_input, rnn_hidden)

        # fc_in = torch.cat(torch.unbind(rnn_hidden[0],dim=0), dim=1) ## last hidden state

        fc_in = catOverTime(rnn_out, 'max')

        fc_out = self.fc(self.rnn_dropout(fc_in))

        return fc_out


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
        dot_prob = F.softmax(torch.bmm(rnn_out, rnn_last.unsqueeze(2)).squeeze(2), dim=1)
        dot_out = torch.bmm(dot_prob.unsqueeze(1), rnn_out).squeeze(1)

        fc1_in = self.rnn_dropout(dot_out)

        fc1_out = F.relu(self.fc1(fc1_in))
        fc1_out = self.fc1_drop(fc1_out)
        fc2_out = self.fc2(fc1_out)

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
        attn_prob = F.softmax(torch.bmm(torch.bmm(rnn_out, attn_bM), rnn_last.unsqueeze(2)).squeeze(2), dim=1)
        attn_out = torch.bmm(attn_prob.unsqueeze(1), rnn_out).squeeze(1)

        attn_out = self.fc_dropout(attn_out)

        fc_out = self.fc(attn_out)

        return fc_out


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

        if self.params['ranking_loss']:
            self.output_layer = ranking_layer(self.rnn_hidden_dim,
                                              targ_size)
        else:
            self.output_layer = F.log_softmax(nn.Linear(self.rnn_hidden_dim, targ_size), dim=1)

    def forward(self, *tensor_feats):

        batch_size = tensor_feats[0].shape[0]

        word_embed_input = self.word_embeddings(tensor_feats[0])

        rnn_input = self.input_dropout(word_embed_input)

        rnn_hidden = self.init_rnn_hidden(batch_size,
                                          self.rnn_hidden_dim,
                                          num_layer=self.params['rnn_layer'])

        rnn_out, rnn_hidden = self.rnn(rnn_input, rnn_hidden)

        # fc_in = torch.cat(torch.unbind(rnn_hidden[0],dim=0), dim=1) ## last hidden state

        rnn_out = self.rnn_dropout(catOverTime(rnn_out, 'max'))

        model_out = self.output_layer(rnn_out)

        return model_out


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

        self.attn_W = torch.nn.Parameter(torch.empty(self.rnn_hidden_dim // 2).uniform_())

        self.rnn_dropout = nn.Dropout(p=self.params['rnn_dropout'])

        self.fc_dropout = nn.Dropout(p=self.params['fc1_dropout'])

        if self.params['ranking_loss']:
            self.output_layer = ranking_layer(self.rnn_hidden_dim // 2,
                                              targ_size)
        else:
            self.output_layer = F.log_softmax(nn.Linear(self.rnn_hidden_dim // 2, targ_size), dim=1)

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
        attn_prob = F.softmax(attn_alpha.squeeze(2), dim=1)
        attn_out = F.tanh(torch.bmm(attn_prob.unsqueeze(1), rnn_out))

        attn_out = self.fc_dropout(attn_out.squeeze(1))

        model_out = self.output_layer(attn_out)

        return model_out


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

        self.fc1_dropout = nn.Dropout(p=self.params['fc1_dropout'])

        if self.params['ranking_loss']:
            self.output_layer = ranking_layer(self.params['fc1_hidden_dim'],
                                              targ_size)
        else:
            self.output_layer = F.log_softmax(nn.Linear(self.params['fc1_hidden_dim'], targ_size), dim=1)

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

        e1_dot_prob = F.softmax(torch.bmm(rnn_out, e1_hidden.unsqueeze(2)).squeeze(2), dim=1)
        e1_dot_out = torch.bmm(e1_dot_prob.unsqueeze(1), rnn_out).squeeze(1)

        e2_dot_prob = F.softmax(torch.bmm(rnn_out, e2_hidden.unsqueeze(2)).squeeze(2), dim=1)
        e2_dot_out = torch.bmm(e2_dot_prob.unsqueeze(1), rnn_out).squeeze(1)

        fc1_in = torch.cat((e1_dot_out, e2_dot_out), dim=1)

        fc1_out = F.relu(self.fc1(fc1_in))
        fc1_out = self.fc1_dropout(fc1_out)
        fc2_out = self.fc2(fc1_out)

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

        self.attn_M = torch.nn.Parameter(torch.empty(self.rnn_hidden_dim, self.rnn_hidden_dim).uniform_())

        self.attn_dropout = nn.Dropout(p=self.params['attn_dropout'])

        self.fc1 = nn.Linear(self.rnn_hidden_dim * 2, self.params['fc1_hidden_dim'])

        self.fc1_dropout = nn.Dropout(p=self.params['fc1_dropout'])

        if self.params['ranking_loss']:
            self.output_layer = ranking_layer(self.params['fc1_hidden_dim'],
                                              targ_size)
        else:
            self.output_layer = F.log_softmax(nn.Linear(self.params['fc1_hidden_dim'], targ_size), dim=1)

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

        e1_mat_prob = F.softmax(torch.bmm(rnn_out_mat, e1_hidden.unsqueeze(2)).squeeze(2), dim=1)
        e1_mat_out = torch.bmm(e1_mat_prob.unsqueeze(1), rnn_out).squeeze(1)

        e2_mat_prob = F.softmax(torch.bmm(rnn_out_mat, e2_hidden.unsqueeze(2)).squeeze(2), dim=1)
        e2_mat_out = torch.bmm(e2_mat_prob.unsqueeze(1), rnn_out).squeeze(1)

        attn_out = self.attn_dropout(torch.cat((e1_mat_out, e2_mat_out), dim=1))

        fc1_out = self.fc1_dropout(F.relu(self.fc1(attn_out)))

        model_out = self.output_layer(fc1_out)

        return model_out


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
        fc2_out = self.fc2(fc1_out)

        return fc2_out