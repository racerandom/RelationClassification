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


def normalize_batch_vector(batch_vector, p=2):
    norm = batch_vector.norm(p=p, dim=1, keepdim=True)
    return batch_vector.div(norm)


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


class baseConfig():

    def __init__(self, word_size, targ_size,
                 max_sent_len, **params):
        # super(baseNN, self).__init__()
        self.params = params
        self.rnn_hidden_dim = self.params['rnn_hidden_dim']
        self.word_size = word_size
        self.targ_size = targ_size
        self.max_sent_len = max_sent_len

    def init_rnn_hidden(self, batch_size, hidden_dim, num_layer=1, bidirectional=True):
        bi_num = 2 if bidirectional else 1
        return (torch.zeros(num_layer * bi_num, batch_size, hidden_dim // bi_num).to(device),
                torch.zeros(num_layer * bi_num, batch_size, hidden_dim // bi_num).to(device))

    # def attn_dot_input(self, *tensor_feats):
    #
    #     batch_size = tensor_feats[0].shape[0]
    #
    #     word_embed_M = self.word_embeddings(tensor_feats[0]) # (batch, max_sent, word_dim)
    #
    #     e1_embed = batch_entity_hidden(word_embed_M, tensor_feats[1]) # (batch, word_dim)
    #     e2_embed = batch_entity_hidden(word_embed_M, tensor_feats[2]) # (batch, word_dim)
    #
    #     e1_dot_weight = F.softmax(torch.bmm(word_embed_M, e1_embed.unsqueeze(2)).squeeze(), dim=1)
    #     e2_dot_weight = F.softmax(torch.bmm(word_embed_M, e2_embed.unsqueeze(2)).squeeze(), dim=1)
    #     dot_weight = (e1_dot_weight + e2_dot_weight) /2 # average weights
    #     attn_applied_M = torch.bmm(dot_weight.unsqueeze(1), word_embed_M).squeeze()
    #
    #     return attn_applied_M


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

    def __init__(self, model_out_dim, targ_size, omit_other=True):
        super(ranking_layer, self).__init__()
        self.targ_size = targ_size
        self.targ_dim = model_out_dim
        r = torch.sqrt(torch.tensor(6 / (targ_size + model_out_dim)))
        self.targ_weight = torch.nn.Parameter(torch.empty(model_out_dim,
                                                          targ_size - 1 if omit_other else targ_size
                                                          ).uniform_(-r, r))

    def forward(self, batch_model_out):

        batch_size = batch_model_out.shape[0]

        batch_pred_scores = torch.einsum('bd,dt->bt', (batch_model_out, self.targ_weight))

        return batch_pred_scores


def get_neg_scores(batch_pred_scores, batch_gold, batch_size, targ_size, omit_other=True):

    batch_neg_score = []
    for i in range(len(batch_pred_scores)):

        mask = torch.ones_like(batch_pred_scores[i], dtype=torch.uint8)
        if not omit_other:
            mask[batch_gold[i]] = 0
        elif batch_gold[i] < targ_size:
            mask[batch_gold[i]] = 0
        neg_scores = batch_pred_scores[i].masked_select(mask)

        neg_score = neg_scores.max(dim=0)[0]
        batch_neg_score.append(neg_score)
    return torch.stack(batch_neg_score)


def ranking_loss(batch_pred_scores, batch_gold, gamma=2., margin_pos=2.5, margin_neg=0.5, omit_other=True):

    batch_size, targ_size = batch_pred_scores.shape

    if omit_other:
        batch_pos_score = torch.stack([torch.tensor(2.5).to(device) if index == 18 else scores[index]
                                       for scores, index in zip(batch_pred_scores, batch_gold)])
    else:
        batch_pos_score = batch_pred_scores.gather(1, batch_gold.unsqueeze(1)).squeeze(1)  # batch * 1

    batch_neg_score = get_neg_scores(batch_pred_scores,
                                     batch_gold,
                                     batch_size,
                                     targ_size,
                                     omit_other=omit_other)  # batch * 1

    loss_pos = torch.log(1 + torch.exp(gamma * (margin_pos - batch_pos_score)))
    loss_neg = torch.log(1 + torch.exp(gamma * (margin_neg + batch_neg_score)))

    # print(loss_pos.mean(dim=0), loss_neg.mean(dim=0))

    return (loss_pos + loss_neg).mean(dim=0)


def infer_pred(batch_pred_scores, omit_other=True):
    if omit_other:
        batch_pred = []
        for scores in batch_pred_scores:
            if scores.max() < 0:
                batch_pred.append(torch.tensor(18).to(device))
            else:
                batch_pred.append(torch.argmax(scores))
        return torch.stack(batch_pred)
    else:
        return torch.argmax(batch_pred_scores, dim=1)


class softmax_layer(nn.Module):

    def __init__(self, model_out_dim, targ_size):
        super(softmax_layer, self).__init__()
        self.fc = nn.Linear(model_out_dim, targ_size)

    def forward(self, model_out):
        fc_out = self.fc(model_out)
        softmax_out = F.log_softmax(fc_out, dim=1)
        return softmax_out


class baseRNN(baseConfig, nn.Module):

    def __init__(self, word_size, targ_size,
                 max_sent_len, max_sdp_len, pre_embed, **params):

        baseConfig.__init__(
            self, word_size, targ_size,
            max_sent_len, **params
        )
        nn.Module.__init__(self)

        if isinstance(pre_embed, np.ndarray):
            self.word_dim = pre_embed.shape[1]
            self.word_embeddings = REData.pre2embed(pre_embed, freeze_mode=self.params['freeze_mode'])

        self.input_dropout = nn.Dropout(p=self.params['input_dropout'])

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
            self.output_layer = softmax_layer(self.rnn_hidden_dim,
                                              targ_size)

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


class DSDPRNN(baseConfig, nn.Module):

    def __init__(self, word_size, dsdp_size, targ_size,
                 max_sent_len, max_sdp_len, pre_embed, **params):

        baseConfig.__init__(
            self, word_size, targ_size,
            max_sent_len, **params
        )
        nn.Module.__init__(self)

        if isinstance(pre_embed, np.ndarray):
            self.word_dim = pre_embed.shape[1]
            self.word_embeddings = REData.pre2embed(pre_embed, freeze_mode=self.params['freeze_mode'])

        self.dsdp_embeddings = nn.Embedding(dsdp_size, self.params['dsdp_dim'], padding_idx=0)

        self.input_dropout = nn.Dropout(p=self.params['input_dropout'])

        self.rnn_input_dim = self.word_dim + 2 * self.params['dsdp_dim']

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
            self.output_layer = softmax_layer(self.rnn_hidden_dim,
                                              targ_size)

    def forward(self, *tensor_feats):

        batch_size = tensor_feats[0].shape[0]

        rnn_input = self.input_dropout(torch.cat((self.word_embeddings(tensor_feats[0]),
                                                  self.dsdp_embeddings(tensor_feats[-2]),
                                                  self.dsdp_embeddings(tensor_feats[-1])
                                                  ), dim=-1))

        rnn_hidden = self.init_rnn_hidden(batch_size,
                                          self.rnn_hidden_dim,
                                          num_layer=self.params['rnn_layer'])

        rnn_out, rnn_hidden = self.rnn(rnn_input, rnn_hidden)

        # fc_in = torch.cat(torch.unbind(rnn_hidden[0],dim=0), dim=1) ## last hidden state

        rnn_out = self.rnn_dropout(catOverTime(rnn_out, 'max'))

        model_out = self.output_layer(rnn_out)

        return model_out



class tokenSDP(nn.Module):

    def __init__(self, word_dim, max_sdp_len, **params):

        nn.Module.__init__(self)

        self.input_dropout = nn.Dropout(p=self.params['input_dropout'])

        self.cnn = nn.Conv1d(word_dim,
                             params['sdp_filter_nb'],
                             params['sdp_kernel_len'])

        kernel_dim = max_sdp_len - params['sdp_kernel_len'] + 1

        self.pool = nn.MaxPool1d(kernel_dim)

        self.cnn_dropout = nn.Dropout(p=params['sdp_cnn_droprate'])

        self.fc = nn.Linear(params['sdp_filter_nb'], params['sdp_fc_dim'])

        self.fc_dropout = nn.Dropout(p=params['sdp_fc_droprate'])

    def forward(self, embed_input):

        cnn_out = F.relu(self.cnn(self.input_dropout(embed_input).transpose(1, 2)))

        pool_out = self.cnn_dropout(self.pool(cnn_out).squeeze(-1))

        fc_out = self.fc_dropout(F.relu(self.fc(pool_out)))

        return fc_out


class TSDPRNN(baseConfig, nn.Module):

    def __init__(self, word_size, targ_size,
                 max_sent_len, max_sdp_len, pre_embed, **params):

        baseConfig.__init__(
            self, word_size, targ_size,
            max_sent_len, **params
        )
        nn.Module.__init__(self)

        if isinstance(pre_embed, np.ndarray):
            self.word_dim = pre_embed.shape[1]
            self.word_embeddings = REData.pre2embed(pre_embed, freeze_mode=self.params['freeze_mode'])

        self.sdp_cnn = tokenSDP(
            self.word_dim, max_sdp_len, **params
        )

        self.input_dropout = nn.Dropout(p=self.params['input_dropout'])

        self.rnn_input_dim = self.word_dim + 2 * self.params['sdp_fc_dim']

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
            self.output_layer = softmax_layer(self.rnn_hidden_dim,
                                              targ_size)

    def forward(self, *tensor_feats):

        batch_size = tensor_feats[0].shape[0]

        word_embed_input = self.word_embeddings(tensor_feats[0])

        tsdp_e1_input = self.word_embeddings(tensor_feats[-2].view(batch_size * self.max_sent_len, -1))
        tsdp_e2_input = self.word_embeddings(tensor_feats[-1].view(batch_size * self.max_sent_len, -1))

        tsdp_e1 = self.sdp_cnn(self.input_dropout(tsdp_e1_input))
        tsdp_e2 = self.sdp_cnn(self.input_dropout(tsdp_e2_input))

        rnn_input = torch.cat((self.input_dropout(word_embed_input),
                               tsdp_e1.view(batch_size, self.max_sent_len, -1),
                               tsdp_e2.view(batch_size, self.max_sent_len, -1)), dim=-1)

        rnn_hidden = self.init_rnn_hidden(batch_size,
                                          self.rnn_hidden_dim,
                                          num_layer=self.params['rnn_layer'])

        rnn_out, rnn_hidden = self.rnn(rnn_input, rnn_hidden)

        # fc_in = torch.cat(torch.unbind(rnn_hidden[0],dim=0), dim=1) ## last hidden state

        rnn_out = self.rnn_dropout(catOverTime(rnn_out, 'max'))

        model_out = self.output_layer(rnn_out)

        return model_out


class attnRNN(baseConfig, nn.Module):

    def __init__(self, word_size, targ_size,
                 max_sent_len, max_sdp_len, pre_embed, **params):

        baseConfig.__init__(
            self, word_size, targ_size,
            max_sent_len, **params
        )
        nn.Module.__init__(self)

        if isinstance(pre_embed, np.ndarray):
            self.word_dim = pre_embed.shape[1]
            self.word_embeddings = REData.pre2embed(pre_embed, freeze_mode=self.params['freeze_mode'])

        self.input_dropout = nn.Dropout(p=self.params['input_dropout'])

        self.rnn_input_dim = self.word_dim

        self.rnn = nn.LSTM(self.rnn_input_dim,
                           self.rnn_hidden_dim // 2,
                           num_layers=self.params['rnn_layer'],
                           batch_first=True,
                           bidirectional=True)

        self.attn_W = torch.nn.Parameter(torch.empty(self.rnn_hidden_dim // 2).uniform_(-0.1, 0.1))

        self.rnn_dropout = nn.Dropout(p=self.params['rnn_dropout'])

        self.fc_dropout = nn.Dropout(p=self.params['fc1_dropout'])

        if self.params['ranking_loss']:
            self.output_layer = ranking_layer(self.rnn_hidden_dim // 2,
                                              targ_size)
        else:
            self.output_layer = softmax_layer(self.rnn_hidden_dim // 2, targ_size)

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

        # attn_bW = self.attn_W.repeat(batch_size, 1)
        # attn_alpha = torch.bmm(rnn_out, attn_bW.unsqueeze(2)).squeeze(2)

        attn_alpha = torch.einsum('bsd,d->bs', (rnn_out, self.attn_W))

        attn_prob = F.softmax(attn_alpha, dim=1)
        attn_out = F.tanh(torch.bmm(attn_prob.unsqueeze(1), rnn_out))

        attn_out = self.fc_dropout(attn_out.squeeze(1))

        model_out = self.output_layer(attn_out)

        return model_out


class attnInBaseRNN(baseConfig, nn.Module):

    def __init__(self, word_size, targ_size,
                 max_sent_len, max_sdp_len, pre_embed, **params):

        baseConfig.__init__(
            self, word_size, targ_size,
            max_sent_len, **params
        )
        nn.Module.__init__(self)

        if isinstance(pre_embed, np.ndarray):
            self.word_dim = pre_embed.shape[1]
            self.word_embeddings = REData.pre2embed(pre_embed, freeze_mode=self.params['freeze_mode'])

        self.input_dropout = nn.Dropout(p=self.params['input_dropout'])

        self.rnn_input_dim = self.word_dim

        self.attn_input = dotAttn()

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
            self.output_layer = softmax_layer(self.rnn_hidden_dim,
                                              targ_size)

    def forward(self, *tensor_feats):

        batch_size = tensor_feats[0].shape[0]

        word_embed_input = self.word_embeddings(tensor_feats[0])

        attn_weight = self.attn_input(*(word_embed_input, tensor_feats[1], tensor_feats[2]))

        attn_applied_input = attn_weight.unsqueeze(2) * word_embed_input

        rnn_input = self.input_dropout(attn_applied_input)

        rnn_hidden = self.init_rnn_hidden(batch_size,
                                          self.rnn_hidden_dim,
                                          num_layer=self.params['rnn_layer'])

        rnn_out, rnn_hidden = self.rnn(rnn_input, rnn_hidden)

        rnn_out = self.rnn_dropout(catOverTime(rnn_out, 'max'))

        model_out = self.output_layer(rnn_out)

        return model_out


class attnDotRNN(baseConfig, nn.Module):

    def __init__(self, word_size, targ_size,
                 max_sent_len, max_sdp_len, pre_embed, **params):

        baseConfig.__init__(
            self, word_size, targ_size,
            max_sent_len, **params
        )
        nn.Module.__init__(self)

        if isinstance(pre_embed, np.ndarray):
            self.word_dim = pre_embed.shape[1]
            self.word_embeddings = REData.pre2embed(pre_embed, freeze_mode=self.params['freeze_mode'])

        self.input_dropout = nn.Dropout(p=self.params['input_dropout'])

        self.rnn_input_dim = self.word_dim

        self.rnn = nn.LSTM(self.rnn_input_dim,
                           self.rnn_hidden_dim // 2,
                           num_layers=self.params['rnn_layer'],
                           batch_first=True,
                           bidirectional=True)

        self.rnn_dropout = nn.Dropout(p=self.params['rnn_dropout'])

        self.fc1_drop = nn.Dropout(p=self.params['fc1_dropout'])

        if self.params['ranking_loss']:
            self.output_layer = ranking_layer(self.rnn_hidden_dim,
                                              targ_size)
        else:
            self.output_layer = softmax_layer(self.rnn_hidden_dim,
                                              targ_size)

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

        attn_out = self.rnn_dropout(dot_out)

        model_out = self.output_layer(attn_out)

        return model_out


class attnMatRNN(baseConfig, nn.Module):

    def __init__(self, word_size, targ_size,
                 max_sent_len, max_sdp_len, pre_embed, **params):

        baseConfig.__init__(
            self, word_size, targ_size,
            max_sent_len, **params
        )
        nn.Module.__init__(self)

        if isinstance(pre_embed, np.ndarray):
            self.word_dim = pre_embed.shape[1]
            self.word_embeddings = REData.pre2embed(pre_embed, freeze_mode=self.params['freeze_mode'])

        self.input_dropout = nn.Dropout(p=self.params['input_dropout'])

        self.rnn_input_dim = self.word_dim

        self.rnn = nn.LSTM(self.rnn_input_dim,
                           self.rnn_hidden_dim // 2,
                           num_layers=self.params['rnn_layer'],
                           batch_first=True,
                           bidirectional=True)

        self.attn_M = torch.nn.Parameter(torch.empty(self.rnn_hidden_dim, self.rnn_hidden_dim).uniform_())

        self.rnn_dropout = nn.Dropout(p=self.params['rnn_dropout'])

        self.fc_dropout = nn.Dropout(p=self.params['fc1_dropout'])

        if self.params['ranking_loss']:
            self.output_layer = ranking_layer(self.rnn_hidden_dim,
                                              targ_size)
        else:
            self.output_layer = softmax_layer(self.rnn_hidden_dim,
                                              targ_size)

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

        # attn_bM = self.attn_M.repeat(batch_size, 1, 1)
        # attn_alpha = torch.bmm(torch.bmm(rnn_out, attn_bM), rnn_last.unsqueeze(2)).squeeze(2)

        attn_alpha = torch.einsum('bsd,de,be->bs', (rnn_out, self.attn_M, rnn_last))

        attn_prob = F.softmax(attn_alpha, dim=1)
        attn_out = torch.bmm(attn_prob.unsqueeze(1), rnn_out).squeeze(1)

        attn_out = self.fc_dropout(attn_out)

        model_out = self.output_layer(attn_out)

        return model_out


class entiAttnDotRNN(baseConfig, nn.Module):

    def __init__(self, word_size, targ_size,
                 max_sent_len, max_sdp_len, pre_embed, **params):

        baseConfig.__init__(
            self, word_size, targ_size,
            max_sent_len, **params
        )
        nn.Module.__init__(self)

        if isinstance(pre_embed, np.ndarray):
            self.word_dim = pre_embed.shape[1]
            self.word_embeddings = REData.pre2embed(pre_embed, freeze_mode=self.params['freeze_mode'])

        self.input_dropout = nn.Dropout(p=self.params['input_dropout'])

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
            self.output_layer = softmax_layer(self.params['fc1_hidden_dim'],
                                              targ_size)

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

        attn_out = torch.cat((e1_dot_out, e2_dot_out), dim=1)

        fc1_out = self.fc1_dropout(F.relu(self.fc1(attn_out)))

        model_out = self.output_layer(fc1_out)

        return model_out


class entiAttnMatRNN(baseConfig, nn.Module):

    def __init__(self, word_size, targ_size,
                 max_sent_len, max_sdp_len, pre_embed, **params):

        baseConfig.__init__(
            self, word_size, targ_size,
            max_sent_len, **params
        )
        nn.Module.__init__(self)

        if isinstance(pre_embed, np.ndarray):
            self.word_dim = pre_embed.shape[1]
            self.word_embeddings = REData.pre2embed(pre_embed, freeze_mode=self.params['freeze_mode'])

        self.input_dropout = nn.Dropout(p=self.params['input_dropout'])

        self.rnn_input_dim = self.word_dim

        self.rnn = nn.LSTM(self.rnn_input_dim,
                           self.rnn_hidden_dim // 2,
                           num_layers=self.params['rnn_layer'],
                           batch_first=True,
                           bidirectional=True)

        self.rnn_dropout = nn.Dropout(p=self.params['rnn_dropout'])

        self.attn_e1_M = torch.nn.Parameter(torch.empty(self.rnn_hidden_dim, self.rnn_hidden_dim).uniform_(-0.1, 0.1))

        self.attn_e2_M = torch.nn.Parameter(torch.empty(self.rnn_hidden_dim, self.rnn_hidden_dim).uniform_(-0.1, 0.1))

        self.attn_dropout = nn.Dropout(p=self.params['attn_dropout'])

        self.fc1 = nn.Linear(self.rnn_hidden_dim * 2, self.params['fc1_hidden_dim'])

        self.fc1_dropout = nn.Dropout(p=self.params['fc1_dropout'])

        if self.params['ranking_loss']:
            self.output_layer = ranking_layer(self.params['fc1_hidden_dim'],
                                              targ_size)
        else:
            self.output_layer = softmax_layer(self.params['fc1_hidden_dim'],
                                              targ_size)

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

        # attn_bM = self.attn_M.repeat(batch_size, 1, 1)

        # rnn_out_mat = torch.bmm(rnn_out, attn_bM)

        # e1_mat_prob = F.softmax(torch.bmm(rnn_out_mat, e1_hidden.unsqueeze(2)).squeeze(2), dim=1)
        # e1_mat_out = torch.bmm(e1_mat_prob.unsqueeze(1), rnn_out).squeeze(1)
        #
        # e2_mat_prob = F.softmax(torch.bmm(rnn_out_mat, e2_hidden.unsqueeze(2)).squeeze(2), dim=1)
        # e2_mat_out = torch.bmm(e2_mat_prob.unsqueeze(1), rnn_out).squeeze(1)

        e1_alpha = F.softmax(torch.einsum('bsd,de,be->bs', (rnn_out, self.attn_e1_M, e1_hidden)))
        e1_mat_out = torch.einsum('bs,bsd->bd', (e1_alpha, rnn_out))

        e2_alpha = F.softmax(torch.einsum('bsd,de,be->bs', (rnn_out, self.attn_e2_M, e2_hidden)))
        e2_mat_out = torch.einsum('bs,bsd->bd', (e2_alpha, rnn_out))

        attn_out = self.attn_dropout(torch.cat((e1_mat_out, e2_mat_out), dim=1))

        fc1_out = self.fc1_dropout(F.relu(self.fc1(attn_out)))

        model_out = self.output_layer(fc1_out)

        return model_out


class mulEntiAttnDotRNN(baseConfig, nn.Module):

    def __init__(self, word_size, targ_size,
                 max_sent_len, max_sdp_len, pre_embed, **params):

        baseConfig.__init__(
            self, word_size, targ_size,
            max_sent_len, **params
        )
        nn.Module.__init__(self)

        if isinstance(pre_embed, np.ndarray):
            self.word_dim = pre_embed.shape[1]
            self.word_embeddings = REData.pre2embed(pre_embed, freeze_mode=self.params['freeze_mode'])

        self.input_dropout = nn.Dropout(p=self.params['input_dropout'])

        self.rnn_input_dim = self.word_dim

        self.rnn = nn.LSTM(self.rnn_input_dim,
                           self.rnn_hidden_dim // 2,
                           num_layers=self.params['rnn_layer'],
                           batch_first=True,
                           bidirectional=True)

        self.rnn_dropout = nn.Dropout(p=self.params['rnn_dropout'])

        self.fc1 = nn.Linear(self.rnn_hidden_dim, self.params['fc1_hidden_dim'])

        self.fc1_drop = nn.Dropout(p=self.params['fc1_dropout'])

        if self.params['ranking_loss']:
            self.output_layer = ranking_layer(self.params['fc1_hidden_dim'],
                                              targ_size)
        else:
            self.output_layer = softmax_layer(self.params['fc1_hidden_dim'],
                                              targ_size)

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

        fc1_out = self.fc1_drop(F.relu(self.fc1(dot_out)))

        model_out = self.output_layer(fc1_out)

        return model_out