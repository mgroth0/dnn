from mlib.boot.mlog import err
def reduced_label(label):
    if label == 'dog' or (label.startswith('S') and len(label) in [2, 3, 4]):
        return 0
    elif label == 'cat' or (label.startswith('N') and len(label) in [3, 4, 5]):
        return 1
    else:
        err('do not know reduced label for ' + label)
