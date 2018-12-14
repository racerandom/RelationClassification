# coding=utf-8
import warnings
import logging
import time
import random

import torch
import torch.nn.functional as F
import torch.utils.data as Data

from sklearn.metrics import classification_report
from statistics import mean, median, variance, stdev

import REUtils
import REModule
import ModuleOptim
import REData
import REEvaluation
warnings.simplefilter("ignore", UserWarning)

REUtils.setup_stream_logger('REOptimize', level=logging.INFO)
logger = logging.getLogger('REOptimize')



device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print('device:', device)

seed = 13
random.seed(seed)

torch_seed = 1337
torch.manual_seed(torch_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def best_score(scores, monitor):
    if not scores:
        return None
    else:
        if monitor.endswith('acc'):
            return max(scores)
        elif monitor.endswith('loss'):
            return min(scores)
        else:
            raise Exception('[ERROR] Unknown monitor mode...')

def eval_data(model, feats, target, rel_idx):
    model.eval()

    with torch.no_grad():
        out = model(*feats)
        loss = F.nll_loss(out, target).item()

        pred = torch.argmax(out, dim=1)
        acc = (pred == target).sum().item() / float(pred.numel())

        idx_set = list(set([p.item() for p in pred]).union(set([t.item() for t in target])))

        logger.info('-' * 80)
        logger.info(classification_report(pred,
                                    target,
                                    target_names=[key for key, value in rel_idx.items() if value in idx_set]
                                    )
              )
        logger.info("test performance: loss %.4f, accuracy %.4f" % (loss, acc))



def model_instance(word_size, targ_size,
                 max_sent_len, pre_embed, **params):

    model = getattr(REModule, params['classification_model'])(
        word_size, targ_size,
        max_sent_len, pre_embed, **params
    ).to(device=device)

    optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=params['lr'],
                                     weight_decay=params['weight_decay'])

    logger.debug(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.debug('* %s' % name)
        else:
            logger.debug('%s' % name)

    logger.debug("Parameters: %i" % ModuleOptim.count_parameters(model))

    return model, optimizer

def optimize_model(train_file, val_file, test_file, embed_file, param_space, max_evals=10):

    monitor = param_space['monitor'][0]

    global_best_checkpoint_file = "models/best_global_%s_checkpoint.pth" % param_space['classification_model'][0]

    word2ix, targ2ix, max_sent_len = REData.generate_feat2ix(train_file)

    train_dataset = REData.generate_data(train_file,
                                         word2ix,
                                         targ2ix,
                                         max_sent_len)

    val_datset = REData.generate_data(val_file,
                                      word2ix,
                                      targ2ix,
                                      max_sent_len)

    test_datset = REData.generate_data(test_file,
                                       word2ix,
                                       targ2ix,
                                       max_sent_len)

    embed_weights = REData.load_pickle(embed_file)

    monitor_score_history, test_loss_history, test_acc_history, params_history = [], [], [], []

    for eval_i in range(1, max_evals + 1):
        params = {}
        for key, values in param_space.items():
            params[key] = random.choice(values)

        logger.info('[Selected %i Params]: %s' % (eval_i, params))

        model, optimizer = model_instance(len(word2ix), len(targ2ix),
                                          max_sent_len, embed_weights, **params)

        global_best_score = best_score(monitor_score_history, monitor)

        local_monitor_score, local_test_loss, local_test_acc = train_model(
            model, optimizer,
            global_best_score,
            train_dataset,
            val_datset,
            test_datset,
            targ2ix,
            **params
        )

        monitor_score_history.append(local_monitor_score)
        test_loss_history.append(local_test_loss)
        test_acc_history.append(local_test_acc)
        params_history.append(params)

        logger.info("Current local %s: %.4f, test acc: %.4f" % (monitor,
                                                                monitor_score_history[-1],
                                                                test_acc_history[-1]))

        best_index = monitor_score_history.index(best_score(monitor_score_history, monitor))
        logger.info("Current best %s: %.4f, test acc: %.4f\n" % (monitor,
                                                                 monitor_score_history[best_index],
                                                                 test_acc_history[best_index]))

    global_best_checkpoint = torch.load(global_best_checkpoint_file,
                                        map_location=lambda storage,
                                        loc: storage)

    # logger.info("test_acc, mean: %.4f, stdev: %.4f" % (mean(test_acc_history), stdev(test_acc_history)))
    logger.info("Final best %s: %.4f, test_acc: %.4f" % (monitor,
                                                         global_best_checkpoint['best_score'],
                                                         global_best_checkpoint['test_acc']))
    logger.info("Final best params: %s" % global_best_checkpoint['params'])

    params = global_best_checkpoint['params']

    model = getattr(REModule, params['classification_model'])(
                len(word2ix), len(targ2ix),
                max_sent_len, embed_weights, **params
            ).to(device=device)

    test_datset = ModuleOptim.batch_to_device(test_datset, device)

    model.load_state_dict(global_best_checkpoint['state_dict'])
    eval_data(model, test_datset[:-1], test_datset[-1], targ2ix)


def train_model(model, optimizer, global_best_score, train_data, val_data, test_data, targ2ix, **params):

    monitor = params['monitor']

    # train_dataset = ModuleOptim.MultipleDatasets(*train_data)

    train_dataset = ModuleOptim.ListDatasets(*train_data)

    train_data_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=params['batch_size'],
        collate_fn=ModuleOptim.collate_fn,
        shuffle=True,
        num_workers=1,
    )


    val_data = ModuleOptim.batch_to_device(val_data, device)
    val_feats = val_data[:-1]
    val_targ = val_data[-1]

    test_data = ModuleOptim.batch_to_device(test_data, device)
    test_feats = test_data[:-1]
    test_targ = test_data[-1]

    val_losses, val_acces, val_f1es, test_losses, test_acces, test_f1es = [], [], [], [], [], []

    patience = params['patience']

    for epoch in range(1, params['epoch_num'] + 1):

        logger.info("epoch %i is runing" % epoch)

        epoch_losses = []
        epoch_acces = []

        for step, train_batch in enumerate(train_data_loader):

            start_time = time.time()

            train_batch = ModuleOptim.batch_to_device(train_batch, device)

            train_feats = train_batch[:-1]
            train_targ = train_batch[-1]

            model.train()
            model.zero_grad()

            pred_prob = model(*train_feats)
            loss = F.nll_loss(pred_prob, train_targ)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params['max_norm'])
            optimizer.step()

            epoch_losses.append(loss.data.item())
            train_pred = torch.argmax(pred_prob, dim=1)
            epoch_acces.append((train_pred == train_targ).sum().item() / float(train_pred.numel()))

            model.eval()
            with torch.no_grad():
                val_prob = model(*val_feats)
                val_loss = F.nll_loss(val_prob, val_targ).item()
                val_pred = torch.argmax(val_prob, dim=1)
                val_acc = (val_pred == val_targ).sum().item() / float(val_pred.numel())
                val_f1 = REEvaluation.f1_score(val_pred, val_targ, labels=[v for k, v in targ2ix.items() if k != 'Other'],
                                           average='macro')

                val_losses.append(val_loss)
                val_acces.append(val_acc)
                val_f1es.append(val_f1)

                test_prob = model(*test_feats)
                test_loss = F.nll_loss(test_prob, test_targ).item()
                test_pred = torch.argmax(test_prob, dim=1)
                test_acc = (test_pred == test_targ).sum().item() / float(test_pred.numel())
                test_f1 = REEvaluation.f1_score(test_pred, test_targ, labels=[v for k, v in targ2ix.items() if k != 'Other'],
                                                average='macro')

                test_losses.append(test_loss)
                test_acces.append(test_acc)
                test_f1es.append(test_f1)

            epoch_scores = locals()[monitor + 'es']

            monitor_score = locals()[params['monitor']]

            global_is_best, global_best_score = ModuleOptim.is_best_score(monitor_score, global_best_score, params['monitor'])

            logger.debug(
                'epoch: %2i, time: %4.1fs, '
                'train loss: %.4f, train acc: %.4f | '
                'val loss: %.4f, val acc: %.4f | '
                'test loss: %.4f, test acc: %.4f' % (epoch,
                                                     time.time() - start_time,
                                                     sum(epoch_losses) / float(len(epoch_losses)),
                                                     sum(epoch_acces) / float(len(epoch_acces)),
                                                     val_loss,
                                                     val_acc,
                                                     test_loss,
                                                     test_acc))

            global_save_info = ModuleOptim.save_checkpoint({
                'epoch': epoch,
                'params': params,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'monitor': params['monitor'],
                'best_score': global_best_score,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'test_loss': test_loss,
                'test_acc': test_acc
            }, global_is_best, "models/best_global_%s_checkpoint.pth" % params['classification_model'])

        # if (patience and
        #         len(val_losses) >= patience and
        #         epoch_scores[-patience] == best_score(epoch_scores[-patience:], monitor)):
        #     print('[Early Stopping] patience reached, stopping...')
        #     break

    monitor_scores = locals()[params['monitor'] + 'es']

    best_local_index = monitor_scores.index(best_score(monitor_scores, params['monitor']))

    return monitor_scores[best_local_index], \
           test_losses[best_local_index], \
           test_acces[best_local_index]

def main():

    pi_feat = ''

    train_file = "data/train%s.pkl" % pi_feat
    val_file = "data/val%s.pkl" % pi_feat
    test_file = "data/test%s.pkl" % pi_feat
    embed_file = "data/glove%s.100d.embed" % pi_feat

    param_space = {
        'classification_model': ['attnInBaseRNN'],
        'freeze_mode': [False],
        'input_dropout': [0.3],
        'rnn_hidden_dim': [200],
        'rnn_layer': [1],
        'rnn_dropout': [0.3],
        'fc1_hidden_dim': [100],
        'fc1_dropout': [0.5],
        'batch_size': [10],
        'epoch_num': [200],
        'lr': [1e-0],
        'weight_decay':[1e-5],
        'max_norm': [1],
        'patience': [10],
        'monitor': ['val_f1']
    }

    optimize_model(train_file, val_file, test_file, embed_file, param_space, max_evals=1)

if __name__ == '__main__':
    main()
