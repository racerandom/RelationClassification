# coding=utf-8

import torch
import torch.nn.functional as F

import REData
import REModule
import ModuleOptim

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print('device:', device)

def eval_output(model, test_feats, test_targ, targ2ix, pred_file, answer_file):

    ix2targ = {v: k for k, v in targ2ix.items()}

    model.eval()
    with torch.no_grad():
        pred_score = model(*test_feats)
        loss = F.nll_loss(pred_score, test_targ).item()
        pred_targ = torch.argmax(pred_score, dim=1)
        acc = (pred_targ == test_targ).sum().item() / float(pred_targ.numel())
        print('checkpoint performance: loss %.4f, acc %.4f' % (loss, acc))

    sent_ids = list(range(8001, 10718))

    assert (len(sent_ids) == len(pred_targ) == len(test_targ))

    with open(pred_file, 'w') as pred_fo:
        for s_id, pred in zip(sent_ids, pred_targ):
            pred_fo.write("%i\t%s\n" % (s_id, ix2targ[pred.item()]))

    with open(answer_file, 'w') as answer_fo:
        for s_id, gold in zip(sent_ids, test_targ):
            answer_fo.write("%i\t%s\n" % (s_id, ix2targ[gold.item()]))


def extrinsic_evaluation(checkpoint_file, train_file, test_file, embed_file, pred_file, answer_file):

    word2ix, e1pos2ix, e2pos2ix, targ2ix, max_sent_len = REData.generate_feat2ix(train_file)

    test_datset = REData.generate_data(test_file,
                                       word2ix,
                                       e1pos2ix,
                                       e2pos2ix,
                                       targ2ix,
                                       max_sent_len)

    embed_weights = REData.load_pickle(embed_file)

    checkpoint = torch.load(checkpoint_file,
                            map_location=lambda storage,
                            loc: storage)



    params = checkpoint['params']

    model = getattr(REModule, params['classification_model'])(
        len(word2ix), len(e1pos2ix), len(e2pos2ix), len(targ2ix),
        max_sent_len, embed_weights, **params
    ).to(device=device)

    test_datset = ModuleOptim.batch_to_device(test_datset, device)

    model.load_state_dict(checkpoint['state_dict'])

    eval_output(model, test_datset[:-1], test_datset[-1], targ2ix, pred_file, answer_file)

def main():
    checkpoint_file = "models/best_global_attnMatRNN_checkpoint.pth"
    train_file = "data/train.pkl"
    test_file = "data/test.pkl"
    embed_file = "data/glove.100d.embed"
    pred_file = "outputs/pred.txt"
    answer_file = "outputs/test.txt"

    extrinsic_evaluation(checkpoint_file, train_file, test_file, embed_file, pred_file, answer_file)

if __name__ == '__main__':
    main()