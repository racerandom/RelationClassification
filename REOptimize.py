# coding=utf-8

import warnings
import sys
import logging
import time
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.utils.data as Data

import REModule
import ModuleOptim
import REData
import REEval

warnings.simplefilter("ignore", UserWarning)


def setup_stream_logger(logger_name, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    formatter = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.addHandler(streamHandler)

    return logger


setup_stream_logger('REOptimize', level=logging.DEBUG)
logger = logging.getLogger('REOptimize')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")


seed = 1
random.seed(seed)

torch_seed = 1337
torch.manual_seed(torch_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def get_checkpoint_file(checkpoint_base, monitor, score):
    return "%s_%s_%f.pth" % (checkpoint_base,
                             monitor,
                             score)


def model_instance(word_size, targ_size,
                   max_sent_len, max_sdp_len, pre_embed, **params):

    model = getattr(REModule, params['classification_model'])(
        word_size, targ_size,
        max_sent_len, max_sdp_len, pre_embed, **params
    ).to(device=device)

    optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=params['lr'],
                                     weight_decay=params['weight_decay'])

    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                             lr=params['lr'],
    #                             momentum=0.9,
    #                             weight_decay=params['weight_decay'],
    #                             nesterov=True)

    logger.debug(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.debug('* %s' % name)
        else:
            logger.debug('%s' % name)

    logger.debug("Parameters: %i" % ModuleOptim.count_parameters(model))

    return model, optimizer


def optimize_model(train_file, test_file, embed_file, param_space,
                   pi_feat, sdp_feat, tsdp_feat, max_evals=10):

    print('device:', device)

    monitor = param_space['monitor'][0]

    checkpoint_base = "models/checkpoint_%s_%i" % (param_space['classification_model'][0],
                                                   int(time.time()))

    train_rels = REData.load_pickle(pickle_file=train_file)

    test_rels = REData.load_pickle(pickle_file=test_file)

    word2ix, targ2ix, max_sent_len, max_sdp_len = REData.prepare_feat2ix(train_rels + test_rels)

    train_indice, val_indice = REData.stratified_split_val(train_rels,
                                                           val_rate=0.1,
                                                           n_splits=1,
                                                           random_seed=0)[0]

    train_dataset = REData.prepare_tensors(
        [train_rels[index] for index in train_indice],
        word2ix,
        targ2ix,
        max_sent_len,
        max_sdp_len,
        pi_feat,
        sdp_feat,
        tsdp_feat
    )

    val_dataset = REData.prepare_tensors(
        [train_rels[index] for index in val_indice],
        word2ix,
        targ2ix,
        max_sent_len,
        max_sdp_len,
        pi_feat,
        sdp_feat,
        tsdp_feat
    )

    test_dataset = REData.prepare_tensors(
        test_rels,
        word2ix,
        targ2ix,
        max_sent_len,
        max_sdp_len,
        pi_feat,
        sdp_feat,
        tsdp_feat
    )

    embed_weights = REData.load_pickle(embed_file)

    global_eval_history = defaultdict(list)

    monitor_score_history = global_eval_history['monitor_score']

    params_history = []

    kbest_scores = []

    for eval_i in range(1, max_evals + 1):

        params = {}

        while not params or params in params_history:
            for key, values in param_space.items():
                params[key] = random.choice(values)

        logger.info('[Selected %i Params]: %s' % (eval_i, params))

        model, optimizer = model_instance(len(word2ix), len(targ2ix),
                                          max_sent_len, max_sdp_len, embed_weights, **params)

        # global_best_score = ModuleOptim.get_best_score(global_eval_history['monitor_score'], monitor)

        kbest_scores = train_model(
            model, optimizer, kbest_scores,
            train_dataset, val_dataset, test_dataset,
            targ2ix,
            checkpoint_base,
            **params
        )

        logger.info("Kbest scores: %s" % kbest_scores)

    best_index = 0 if monitor.endswith('loss') else -1

    best_checkpoint_file = get_checkpoint_file(checkpoint_base, monitor, kbest_scores[best_index])

    best_checkpoint = torch.load(best_checkpoint_file,
                                 map_location=lambda storage,
                                 loc: storage)

    # logger.info("test_acc, mean: %.4f, stdev: %.4f" % (mean(test_acc_history), stdev(test_acc_history)))
    logger.info("Final best %s: %.4f" % (monitor,
                                         best_checkpoint['best_score']))
    logger.info("Final best params: %s" % best_checkpoint['params'])

    params = best_checkpoint['params']

    model = getattr(REModule, params['classification_model'])(
                len(word2ix), len(targ2ix),
                max_sent_len, embed_weights, **params
            ).to(device=device)

    test_data_loader = Data.DataLoader(
        dataset=ModuleOptim.CustomizedDatasets(*test_dataset),
        batch_size=64,
        collate_fn=ModuleOptim.collate_fn,
        shuffle=True,
        num_workers=1,
    )

    model.load_state_dict(best_checkpoint['state_dict'])

    loss_func = REModule.ranking_loss if param_space['ranking_loss'][0] else F.nll_loss

    REEval.batch_eval(model,
                      test_data_loader,
                      targ2ix,
                      loss_func,
                      param_space['omit_other'][0],
                      report_result=True)


def train_model(model, optimizer, kbest_scores,
                train_data, val_data, test_data,
                targ2ix, checkpoint_base, **params):

    monitor = params['monitor']

    train_data_loader = Data.DataLoader(
        dataset=ModuleOptim.CustomizedDatasets(*train_data),
        batch_size=params['batch_size'],
        collate_fn=ModuleOptim.collate_fn,
        shuffle=True,
        num_workers=1,
    )

    val_data_loader = Data.DataLoader(
        dataset=ModuleOptim.CustomizedDatasets(*val_data),
        batch_size=params['batch_size'],
        collate_fn=ModuleOptim.collate_fn,
        shuffle=True,
        num_workers=1,
    )

    test_data_loader = Data.DataLoader(
        dataset=ModuleOptim.CustomizedDatasets(*test_data),
        batch_size=params['batch_size'],
        collate_fn=ModuleOptim.collate_fn,
        shuffle=True,
        num_workers=1,
    )

    eval_history = defaultdict(list)

    monitor_score_history = eval_history[params['monitor']]

    patience = params['patience']

    for epoch in range(1, params['epoch_num'] + 1):

        epoch_start_time = time.time()

        epoch_losses = []
        epoch_acces = []

        step_num = len(train_data_loader)

        for step, train_batch in enumerate(train_data_loader):

            start_time = time.time()

            train_batch = ModuleOptim.batch_to_device(train_batch, device)
            train_feats = train_batch[:-1]
            train_targ = train_batch[-1]

            model.train()
            model.zero_grad()

            pred_prob = model(*train_feats)
            if params['ranking_loss']:
                train_loss = REModule.ranking_loss(pred_prob, train_targ,
                                                   gamma=params['gamma'],
                                                   margin_pos=params['margin_pos'],
                                                   margin_neg=params['margin_neg'],
                                                   omit_other=params['omit_other'])
                train_pred = REModule.infer_pred(pred_prob, omit_other=params['omit_other'])
            else:
                train_loss = F.nll_loss(pred_prob, train_targ)
                train_pred = torch.argmax(pred_prob, dim=1)

            train_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params['max_norm'])
            optimizer.step()

            epoch_losses.append(train_loss.item())
            train_acc = (train_pred == train_targ).sum().item() / float(train_pred.numel())
            epoch_acces.append(train_acc)

            if (step != 0 and step % params['check_interval'] == 0) or step == step_num - 1:

                _, _, [val_loss, val_acc, val_f1] = REEval.batch_eval(model,
                                                                      val_data_loader,
                                                                      targ2ix,
                                                                      ranking_loss=params['ranking_loss'],
                                                                      omit_other=params['omit_other']
                                                                      )

                eval_history['val_loss'].append(val_loss)
                eval_history['val_acc'].append(val_acc)
                eval_history['val_f1'].append(val_f1)

                monitor_score = round(locals()[monitor], 6)

                # global_is_best, global_best_score = ModuleOptim.is_best_score(monitor_score,
                #                                                               global_best_score,
                #                                                               monitor)

                is_kbest, kbest_scores = ModuleOptim.update_kbest_scores(kbest_scores,
                                                                         monitor_score,
                                                                         monitor,
                                                                         kbest=params['kbest_checkpoint'])
                # print(kbest_scores)

                if is_kbest and len(kbest_scores) == params['kbest_checkpoint'] + 1:
                    removed_index = -1 if monitor.endswith('loss') else 0
                    removed_score = kbest_scores.pop(removed_index)
                    ModuleOptim.delete_checkpoint(get_checkpoint_file(checkpoint_base,
                                                                      monitor,
                                                                      removed_score))
                    assert len(kbest_scores) == params['kbest_checkpoint']

                global_save_info = ModuleOptim.save_checkpoint({
                    'params': params,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'monitor': monitor,
                    'best_score': monitor_score,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                }, is_kbest, get_checkpoint_file(checkpoint_base,
                                                 monitor,
                                                 monitor_score))

                _, _, [test_loss, test_acc, test_f1] = REEval.batch_eval(model,
                                                                         test_data_loader,
                                                                         targ2ix,
                                                                         ranking_loss=params['ranking_loss'],
                                                                         omit_other=params['omit_other']
                                                                         )

                logger.debug(
                    'epoch: %2i, step: %4i, time: %4.1fs | '
                    'train loss: %.4f, train acc: %.4f | '
                    'val loss: %.4f, val acc: %.4f | '
                    'test loss: %.4f, test acc: %.4f %s'
                    % (
                        epoch,
                        step,
                        time.time() - start_time,
                        train_loss,
                        train_acc,
                        val_loss,
                        val_acc,
                        test_loss,
                        test_acc,
                        global_save_info
                    )
                )

        eval_history['epoch_best'].append(
            ModuleOptim.get_best_score(monitor_score_history[-step_num * params['check_interval']:],
                                       monitor)
        )

        logger.info(
            "epoch %i finished in %.2fs, "
            "train loss: %.4f, train acc: %.4f"
            % (
                epoch,
                time.time() - epoch_start_time,
                sum(epoch_losses) / float(len(epoch_losses)),
                sum(epoch_acces) / float(len(epoch_acces)),
               )
        )

        if (patience and
            len(eval_history['epoch_best']) >= patience and
            eval_history['epoch_best'][-patience] == ModuleOptim.get_best_score(eval_history['epoch_best'][-patience:],
                                                                                monitor)):
            print('[Early Stopping] patience reached, stopping...')
            break

    best_local_index = monitor_score_history.index(ModuleOptim.get_best_score(monitor_score_history, params['monitor']))

    return kbest_scores


def main():

    classification_model = 'attnRNN'

    param_space = {
        'classification_model': [classification_model],
        'freeze_mode': [False],
        'sdp_filter_nb': [100],
        'sdp_kernel_len': [3],
        'sdp_cnn_droprate': [0.5],
        'sdp_fc_dim': [50],
        'sdp_fc_droprate': [0.5],
        'input_dropout': [0.3],
        'rnn_hidden_dim': [200],
        'rnn_layer': [1],
        'rnn_dropout': [0.3],
        'attn_dropout': [0.3],
        'fc1_hidden_dim': [200],
        'fc1_dropout': [0.5],
        'batch_size': [32],
        'epoch_num': [200],
        'lr': [1e-0],
        'weight_decay': [1e-4],
        'max_norm': [3],
        'patience': [10],
        'monitor': ['val_f1'],
        'check_interval': [100],    # checkpoint based on val performance given a step interval
        'kbest_checkpoint': [5],
        'ranking_loss': [True],
        'omit_other': [True],
        'gamma': [1, 1.5, 2],
        'margin_pos': [1.5, 2, 2.5, 3],
        'margin_neg': [0.5, 1],
    }

    # pi_feat = '.PI' if classification_model in ['baseRNN',
    #                                             'attnRNN',
    #                                             'attnDotRNN',
    #                                             'attnMatRNN'] else ''

    pi_feat = True
    sdp_feat = False
    tsdp_feat = False

    feat_suffix = ''
    feat_suffix += '.SDP' if sdp_feat else ''
    feat_suffix += '.TSDP' if tsdp_feat else ''
    feat_suffix += '.PI' if pi_feat else ''

    train_file = "data/train%s.pkl" % feat_suffix
    test_file = "data/test%s.pkl" % feat_suffix
    embed_file = "data/glove.100d.embed"

    optimize_model(train_file, test_file, embed_file, param_space, pi_feat, sdp_feat, tsdp_feat, max_evals=1)


if __name__ == '__main__':
    main()
