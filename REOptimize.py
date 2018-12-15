# coding=utf-8
import warnings
import logging
import time
import random
from collections import defaultdict

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

REUtils.setup_stream_logger('REOptimize', level=logging.DEBUG)
logger = logging.getLogger('REOptimize')



device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print('device:', device)

seed = 1337
random.seed(seed)

torch_seed = 1337
torch.manual_seed(torch_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


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


def batch_eval(model, data_loader, targ2ix, report_result=False):

    model.eval()
    with torch.no_grad():

        pred, targ = torch.LongTensor().to(device), torch.LongTensor().to(device)
        loss_step, acc_step, f1_step = [], [], []

        for batch in data_loader:
            batch = ModuleOptim.batch_to_device(batch, device)
            batch_feats = batch[:-1]
            batch_targ = batch[-1]

            batch_prob = model(*batch_feats)
            batch_loss = F.nll_loss(batch_prob, batch_targ).item()
            batch_pred = torch.argmax(batch_prob, dim=1)
            batch_acc = (batch_pred == batch_targ).sum().item() / float(batch_pred.numel())
            batch_f1 = REEvaluation.f1_score(batch_pred, batch_targ,
                                             labels=[v for k, v in targ2ix.items() if k != 'Other'],
                                             average='macro')

            loss_step.append(batch_loss)
            acc_step.append(batch_acc)
            f1_step.append(batch_f1)

            pred = torch.cat((pred, batch_pred), dim=0)
            targ = torch.cat((targ, batch_targ), dim=0)

        assert pred.shape[0] == targ.shape[0]

        loss = mean(loss_step)
        acc = mean(acc_step)
        f1 = mean(f1_step)

        if report_result:

            idx_set = list(set([p.item() for p in pred]).union(set([t.item() for t in targ])))
            logger.info('-' * 80)
            logger.info(classification_report(pred,
                                              targ,
                                              target_names=[k for k, v in targ2ix.items() if v in idx_set]
                                              )
                        )
            logger.info("test performance: loss %.4f, accuracy %.4f" % (loss, acc))


    return loss, acc, f1


def optimize_model(train_file, val_file, test_file, embed_file, param_space, max_evals=10):

    monitor = param_space['monitor'][0]

    global_checkpoint_file = "models/best_global_%s_%i_checkpoint.pth" % (param_space['classification_model'][0],
                                                                          int(time.time()))

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

    global_eval_history = defaultdict(list)
    monitor_score_history = global_eval_history['monitor_score']

    for eval_i in range(1, max_evals + 1):
        params = {}
        for key, values in param_space.items():
            params[key] = random.choice(values)

        logger.info('[Selected %i Params]: %s' % (eval_i, params))

        model, optimizer = model_instance(len(word2ix), len(targ2ix),
                                          max_sent_len, embed_weights, **params)

        global_best_score = ModuleOptim.get_best_score(global_eval_history['monitor_score'], monitor)

        local_monitor_score, local_val_loss, local_val_acc, local_val_f1 = train_model(
            model, optimizer,
            global_best_score,
            train_dataset,
            val_datset,
            test_datset,
            targ2ix,
            global_checkpoint_file,
            **params
        )

        monitor_score_history.append(local_monitor_score)
        global_eval_history['val_loss'].append(local_val_loss)
        global_eval_history['val_acc'].append(local_val_acc)
        global_eval_history['val_f1'].append(local_val_f1)

        print(local_val_loss, local_val_acc, local_val_f1)

        logger.info("[Monitoring %s]Local val loss: %.4f, val acc: %.4f, val f1: %.4f" % (
            monitor,
            global_eval_history['val_loss'][-1],
            global_eval_history['val_acc'][-1],
            global_eval_history['val_f1'][-1])
        )

        best_index = monitor_score_history.index(ModuleOptim.get_best_score(monitor_score_history,
                                                                            monitor))
        logger.info("[Monitoring %s]Global val loss: %.4f, val acc: %.4f, val f1: %.4f\n" % (
            monitor,
            global_eval_history['val_loss'][best_index],
            global_eval_history['val_acc'][best_index],
            global_eval_history['val_f1'][best_index])
        )

    global_best_checkpoint = torch.load(global_checkpoint_file,
                                        map_location=lambda storage,
                                        loc: storage)

    # logger.info("test_acc, mean: %.4f, stdev: %.4f" % (mean(test_acc_history), stdev(test_acc_history)))
    logger.info("Final best %s: %.4f" % (monitor,
                                         global_best_checkpoint['best_score']))
    logger.info("Final best params: %s" % global_best_checkpoint['params'])

    params = global_best_checkpoint['params']

    model = getattr(REModule, params['classification_model'])(
                len(word2ix), len(targ2ix),
                max_sent_len, embed_weights, **params
            ).to(device=device)

    test_datset = ModuleOptim.batch_to_device(test_datset, device)

    model.load_state_dict(global_best_checkpoint['state_dict'])
    eval_data(model, test_datset[:-1], test_datset[-1], targ2ix)


def train_model(model, optimizer, global_best_score, train_data, val_data, test_data, targ2ix, global_checkpoint_file, **params):

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
            train_loss = F.nll_loss(pred_prob, train_targ)
            train_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params['max_norm'])
            optimizer.step()

            epoch_losses.append(train_loss.item())
            train_pred = torch.argmax(pred_prob, dim=1)
            train_acc = (train_pred == train_targ).sum().item() / float(train_pred.numel())
            epoch_acces.append(train_acc)

            if (step != 0 and step % params['check_interval'] == 0) or step == step_num - 1:

                val_loss, val_acc, val_f1 = batch_eval(model, val_data_loader, targ2ix)

                eval_history['val_loss'].append(val_loss)
                eval_history['val_acc'].append(val_acc)
                eval_history['val_f1'].append(val_f1)

                monitor_score = locals()[monitor]

                global_is_best, global_best_score = ModuleOptim.is_best_score(monitor_score,
                                                                              global_best_score,
                                                                              monitor)

                global_save_info = ModuleOptim.save_checkpoint({
                    'params': params,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'monitor': monitor,
                    'best_score': global_best_score,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                }, global_is_best, global_checkpoint_file)

                test_loss, test_acc, test_f1 = batch_eval(model, test_data_loader, targ2ix)

                logger.debug(
                    'epoch: %2i, step: %4i, time: %4.1fs | '
                    'train loss: %.4f, train acc: %.4f | '
                    'val loss: %.4f, val acc: %.4f | '
                    'test loss: %.4f, test acc: %.4f'
                    % (
                        epoch,
                        step,
                        time.time() - start_time,
                        train_loss,
                        train_acc,
                        val_loss,
                        val_acc,
                        test_loss,
                        test_acc
                    )
                )

        eval_history['epoch_best'].append(ModuleOptim.get_best_score(monitor_score_history[-step_num * params['check_interval']:],
                                                                     monitor))

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
                eval_history['epoch_best'][-patience] == ModuleOptim.get_best_score(eval_history['epoch_best'][-patience:], monitor)):
            print('[Early Stopping] patience reached, stopping...')
            break

    best_local_index = monitor_score_history.index(ModuleOptim.get_best_score(monitor_score_history, params['monitor']))

    return monitor_score_history[best_local_index], \
           eval_history['val_loss'][best_local_index], \
           eval_history['val_acc'][best_local_index], \
           eval_history['val_f1'][best_local_index]


def main():

    pi_feat = ''

    train_file = "data/train%s.pkl" % pi_feat
    val_file = "data/val%s.pkl" % pi_feat
    test_file = "data/test%s.pkl" % pi_feat
    embed_file = "data/glove%s.100d.embed" % pi_feat

    param_space = {
        'classification_model': ['entiAttnMatRNN'],
        'freeze_mode': [False],
        'input_dropout': [0.3],
        'rnn_hidden_dim': range(100, 1000 + 1, 20),
        'rnn_layer': [1],
        'rnn_dropout': [0.3],
        'attn_dropout': [0.3],
        'fc1_hidden_dim': range(100, 1000 + 1, 20),
        'fc1_dropout': [0.5],
        'batch_size': [32],
        'epoch_num': [200],
        'lr': [1e-0],
        'weight_decay':[1e-5],
        'max_norm': [1, 3, 5],
        'patience': [10],
        'monitor': ['val_f1'],
        'check_interval': [20],    # checkpoint based on val performance given a step interval
    }

    optimize_model(train_file, val_file, test_file, embed_file, param_space, max_evals=100)

if __name__ == '__main__':
    main()
