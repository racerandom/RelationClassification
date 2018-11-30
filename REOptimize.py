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
warnings.simplefilter("ignore", UserWarning)

REUtils.setup_stream_logger('REOptimize', level=logging.DEBUG)
logger = logging.getLogger('REOptimize')



device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print('device:', device)

seed = 1336
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

        print('-' * 80)
        print(classification_report(pred,
                                    target,
                                    target_names=[key for key, value in rel_idx.items() if value in idx_set]
                                    )
              )
        print(loss, acc)


def model_instance(word_size, e1pos_size, e2pos_size, targ_size,
                 max_sent_len, pre_embed, **params):

    model = getattr(REModule, params['classification_model'])(
        word_size, e1pos_size, e2pos_size, targ_size,
        max_sent_len, pre_embed, **params
    ).to(device=device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params['lr'])

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

    word2ix, e1pos2ix, e2pos2ix, targ2ix, max_sent_len = REData.generate_feat2ix(train_file)

    train_word, train_e1pos, train_e2pos, train_targs = REData.generate_data(train_file,
                                                                      word2ix,
                                                                      e1pos2ix,
                                                                      e2pos2ix,
                                                                      targ2ix,
                                                                      max_sent_len)

    val_word, val_e1pos, val_e2pos, val_targs = REData.generate_data(val_file,
                                                              word2ix,
                                                              e1pos2ix,
                                                              e2pos2ix,
                                                              targ2ix,
                                                              max_sent_len)

    test_word, test_e1pos, test_e2pos, test_targs = REData.generate_data(test_file,
                                                                  word2ix,
                                                                  e1pos2ix,
                                                                  e2pos2ix,
                                                                  targ2ix,
                                                                  max_sent_len)

    embed_weights = REData.load_pickle(embed_file)


    monitor_scores, test_losses, test_acces, test_params = [], [], [], []


    for eval_i in range(1, max_evals + 1):
        params = {}
        for key, values in param_space.items():
            params[key] = random.choice(values)

        logger.info('[Selected %i Params]: %s' % (eval_i, params))

        model, optimizer = model_instance(len(word2ix), len(e1pos2ix), len(e2pos2ix), len(targ2ix),
                                          max_sent_len, embed_weights, **params)

        global_best_score = best_score(monitor_scores, monitor)

        local_monitor_score, local_test_loss, local_test_acc, local_param = train_model(
            model, optimizer,
            global_best_score,
            (train_word, train_e1pos, train_e2pos, train_targs),
            (val_word, val_e1pos, val_e2pos, val_targs),
            (test_word, test_e1pos, test_e2pos, test_targs),
            targ2ix,
            **params
        )

        monitor_scores.append(local_monitor_score)
        test_losses.append(local_test_loss)
        test_acces.append(local_test_acc)
        test_params.append(local_param)

        best_index = monitor_scores.index(best_score(monitor_scores, monitor))
        logger.info("Current best %s: %.4f, test acc: %.4f\n" % (monitor,
                                                               monitor_scores[best_index],
                                                               test_acces[best_index]))

    logger.info("test_acc, mean: %.4f, stdev: %.4f" % (mean(test_acces), stdev(test_acces)))
    global_best_checkpoint = torch.load(global_best_checkpoint_file,
                                        map_location=lambda storage,
                                        loc: storage)

    logger.info("Final best test_acc: %.4f" % global_best_checkpoint['test_acc'])
    logger.info("Final best params:", global_best_checkpoint['params'])

def train_model(model, optimizer, global_best_score, train_data, val_data, test_data, targ2ix, **params):

    train_dataset = ModuleOptim.MultipleDatasets(*train_data)

    train_data_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=1,
    )

    val_word, val_e1pos, val_e2pos, val_targ = ModuleOptim.batch_to_device(val_data, device)

    test_word, test_e1pos, test_e2pos, test_targ = ModuleOptim.batch_to_device(test_data, device)

    local_best_score = None

    val_losses = []

    patience = params['patience']

    for epoch in range(1, params['epoch_num'] + 1):
        epoch_losses = []
        epoch_acces = []
        start_time = time.time()
        for step, train_sample in enumerate(train_data_loader):
            train_word, train_e1pos, train_e2pos, train_targ = ModuleOptim.batch_to_device(train_sample, device)

            model.train()
            model.zero_grad()

            pred_prob = model(train_word, train_e1pos, train_e2pos)
            loss = F.nll_loss(pred_prob, train_targ)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params['max_norm'])
            optimizer.step()

            epoch_losses.append(loss.data.item())
            train_pred = torch.argmax(pred_prob, dim=1)
            epoch_acces.append((train_pred == train_targ).sum().item() / float(train_pred.numel()))

        model.eval()
        with torch.no_grad():
            val_prob = model(val_word, val_e1pos, val_e2pos)
            val_loss = F.nll_loss(val_prob, val_targ).item()
            val_pred = torch.argmax(val_prob, dim=1)
            val_acc = (val_pred == val_targ).sum().item() / float(val_pred.numel())

            val_losses.append(val_loss)

            test_prob = model(test_word, test_e1pos, test_e2pos)
            test_loss = F.nll_loss(test_prob, test_targ).item()
            test_pred = torch.argmax(test_prob, dim=1)
            test_acc = (test_pred == test_targ).sum().item() / float(test_pred.numel())

        if patience and len(val_losses) >= patience and val_losses[-patience] == min(val_losses[-patience:]):
            print('[Early Stopping] patience reached, stopping...')
            break

        monitor_score = locals()[params['monitor']]

        local_is_best, local_best_score = ModuleOptim.is_best_score(monitor_score, local_best_score, params['monitor'])

        global_is_best, global_best_score = ModuleOptim.is_best_score(monitor_score, global_best_score, params['monitor'])

        logger.info(
            'epoch: %i, time: %.4f, '
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

        local_save_info = ModuleOptim.save_checkpoint({
            'epoch': epoch,
            'params': params,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'monitor': params['monitor'],
            'best_score': local_best_score,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        }, local_is_best, "models/best_local_%s_checkpoint.pth" % params['classification_model'])

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

    local_checkpoint = torch.load("models/best_local_%s_checkpoint.pth" % params['classification_model'],
                                  map_location=lambda storage,
                                  loc: storage)
    logger.info('Local best: val_loss %.4f, val_acc %.4f | test_loss %.4f, test_acc %.4f' % (
        local_checkpoint['val_loss'],
        local_checkpoint['val_acc'],
        local_checkpoint['test_loss'],
        local_checkpoint['test_acc']
    ))

    model.load_state_dict(local_checkpoint['state_dict'])

    eval_data(model, (test_word, test_e1pos, test_e2pos), test_targ, targ2ix)
    return local_best_score, local_checkpoint['test_loss'], local_checkpoint['test_acc'], local_checkpoint['params']

def main():
    train_file = "data/train.pkl"
    val_file = "data/val.pkl"
    test_file = "data/test.pkl"
    embed_file = "data/glove.100d.embed"

    param_space = {
        'classification_model': ['baseRNN'],
        'freeze_mode': [True],
        'pos_dim': range(5, 30 + 1, 5),
        'input_dropout': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
        'rnn_hidden_dim': range(100, 500 + 1, 20),
        'rnn_layer': [1],
        'rnn_dropout': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
        'fc1_hidden_dim': range(100, 500 + 1, 20),
        'fc1_dropout': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
        'batch_size': [16, 32, 64, 128],
        'epoch_num': [50],
        'lr': [0.01, 0.001],
        'max_norm': [1, 5, 10],
        'patience': [10],
        'monitor': ['val_loss']
    }

    optimize_model(train_file, val_file, test_file, embed_file, param_space, max_evals=100)

if __name__ == '__main__':
    main()
