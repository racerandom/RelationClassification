# coding=utf-8

def feat2vocab(feat_list):
    if feat_list:
        vocab = set()
        for line in feat_list:
            for tok in line:
                vocab.add(tok)
        return vocab
    else:
        return None