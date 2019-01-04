# coding=utf-8
import sys
import os
import logging

import torch
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import f1_score, classification_report
# from statistics import mean, median, variance, stdev

import REData
import REModule
import ModuleOptim

logger = logging.getLogger('REOptimize')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")


def batch_eval(model, data_loader, targ2ix, ranking_loss=False, omit_other=False, report_result=False):

    model.eval()
    with torch.no_grad():

        pred_prob, targ = torch.FloatTensor().to(device), torch.LongTensor().to(device)

        for batch in data_loader:
            batch = ModuleOptim.batch_to_device(batch, device)
            batch_feats = batch[:-1]
            batch_targ = batch[-1]

            batch_prob = model(*batch_feats)

            pred_prob = torch.cat((pred_prob, batch_prob), dim=0)
            targ = torch.cat((targ, batch_targ), dim=0)

        assert pred_prob.shape[0] == targ.shape[0]

        if ranking_loss:
            loss = REModule.ranking_loss(pred_prob, targ, omit_other=omit_other).item()
            pred = REModule.infer_pred(pred_prob, omit_other=omit_other)
        else:
            loss = F.nll_loss(pred_prob, targ).item()
            pred = torch.argmax(pred_prob, dim=1)
        acc = (pred == targ).sum().item() / float(pred.numel())
        f1 = f1_score(
            pred, targ,
            labels=[v for k, v in targ2ix.items() if k != 'Other'],
            average='macro'
        )

        if report_result:

            idx_set = list(set([p.item() for p in pred]).union(set([t.item() for t in targ])))
            logger.info('-' * 80)
            logger.info(classification_report(pred,
                                              targ,
                                              target_names=[k for k, v in targ2ix.items() if v in idx_set]
                                              )
                        )
            logger.info("test performance: loss %.4f, accuracy %.4f" % (loss, acc))

    return pred, targ, [loss, acc, f1]


def eval_output(model, test_datset, targ2ix, ranking_loss, omit_other, pred_file, answer_file):

    ix2targ = {v: k for k, v in targ2ix.items()}

    pred, targ, [loss, acc, f1] = batch_eval(model,
                                             test_datset,
                                             targ2ix,
                                             ranking_loss,
                                             omit_other)

    print('checkpoint performance: loss %.4f, acc %.4f, macro-F1 %.4f\n' % (loss, acc, f1))

    sent_ids = list(range(8001, 10718))

    assert (len(sent_ids) == len(pred) == len(targ))

    file_dir = os.path.dirname(pred_file)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    with open(pred_file, 'w') as pred_fo:
        for s_id, pred in zip(sent_ids, pred):
            pred_fo.write("%i\t%s\n" % (s_id, ix2targ[pred.item()]))

    with open(answer_file, 'w') as answer_fo:
        for s_id, gold in zip(sent_ids, targ):
            answer_fo.write("%i\t%s\n" % (s_id, ix2targ[gold.item()]))


def extrinsic_eval(checkpoint_file, train_file, test_file, embed_file, pred_file, answer_file):

    train_rels = REData.load_pickle(pickle_file=train_file)

    test_rels = REData.load_pickle(pickle_file=test_file)

    word2ix, targ2ix, max_sent_len = REData.prepare_feat2ix(train_rels + test_rels)

    test_data = REData.prepare_tensors(
        test_rels,
        word2ix,
        targ2ix,
        max_sent_len
    )

    embed_weights = REData.load_pickle(embed_file)

    checkpoint = torch.load(checkpoint_file,
                            map_location=lambda storage,
                            loc: storage)

    params = checkpoint['params']

    test_data_loader = Data.DataLoader(
        dataset=ModuleOptim.CustomizedDatasets(*test_data),
        batch_size=params['batch_size'],
        collate_fn=ModuleOptim.collate_fn,
        shuffle=True,
        num_workers=1,
    )

    print("[checkpoint] loss %.4f, acc %.4f, f1 %.4f" % (
        checkpoint['val_loss'],
        checkpoint['val_acc'],
        checkpoint['val_f1']
    ))

    model = getattr(REModule, params['classification_model'])(
        len(word2ix), len(targ2ix),
        max_sent_len, embed_weights, **params
    ).to(device=device)

    model.load_state_dict(checkpoint['state_dict'])

    eval_output(model, test_data_loader, targ2ix,
                params['ranking_loss'], params['omit_other'],
                pred_file, answer_file)

    print(params)


def call_official_eval(pred_file, answer_file):
    import subprocess
    eval_out = subprocess.Popen(["data/SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2/"
                                 "semeval2010_task8_scorer-v1.2.pl",
                                 pred_file,
                                 answer_file], stdout=subprocess.PIPE)
    print("\n%s" % eval_out.communicate()[0].decode("utf-8").strip().split('\n')[-1])


def main():
    print('device:', device)

    checkpoint_file = sys.argv[1]

    # pi_feat = '.PI' if checkpoint_file.split('_')[1] in [
    #     'baseRNN',
    #     'attnRNN',
    #     'attnDotRNN',
    #     'attnMatRNN'
    # ] else ''

    pi_feat = False

    train_file = "data/train%s.pkl" % ('.PI' if pi_feat else '')
    test_file = "data/test%s.pkl" % ('.PI' if pi_feat else '')
    embed_file = "data/glove%s.100d.embed" % ('.PI' if pi_feat else '')
    pred_file = "outputs/pred.txt"
    answer_file = "outputs/test.txt"

    extrinsic_eval(checkpoint_file, train_file, test_file, embed_file, pred_file, answer_file)

    call_official_eval(pred_file, answer_file)


if __name__ == '__main__':
    main()