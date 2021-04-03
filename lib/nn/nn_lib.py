import copy

from mlib.boot.stream import ismember, mod, numel, randperm
from mlib.file import Folder



def calc_steps(N_IMAGES, TRAIN_TEST_SPLIT, BATCH_SIZE):
    N_TRAIN_IMAGES = int(N_IMAGES * TRAIN_TEST_SPLIT)
    N_TEST_IMAGES = int(N_IMAGES * (1 - TRAIN_TEST_SPLIT))
    TRAIN_STEPS = int(N_TRAIN_IMAGES / BATCH_SIZE)
    TEST_STEPS = int(N_TEST_IMAGES / BATCH_SIZE)
    return TRAIN_STEPS, TEST_STEPS, N_TRAIN_IMAGES, N_TEST_IMAGES

def symm(im, arg):
    ODD = mod(im.shape[1], 2) > 0
    # err('please use images with an even number of columns (will fix later)')
    n_sym = 0
    sym_i = []
    n_pix = numel(im)
    if ODD:
        left_pix = (n_pix - im.shape[0]) / 2
    else:
        left_pix = n_pix / 2
    left_pix = int(left_pix)
    i = 0
    for x in range(im.shape[1]):
        if x < im.shape[1] / 2:
            x2 = im.shape[1] - (x + 1)
            for y in range(im.shape[0]):
                i = i + 1
                this_sym = (im[y, x] == im[y, x2])
                n_sym = n_sym + this_sym
                if this_sym:
                    sym_i.append(i)

    if arg is None:
        pass
        # what was that for? sym_im was unused
        # sym_im = n_sym / left_pix
    else:
        p_sym = arg
        n_sym_target = round(left_pix * p_sym)
        sym_i_target = copy.deepcopy(sym_i)

        #     get all sym_i_target
        more = n_sym_target - len(sym_i)
        more_get_i = more
        #     while more_get_i > 0
        new_order = randperm(left_pix)
    i = 0
    for new in new_order:
        i = i + 1
        if new not in sym_i:  # and not ismember(new, sym_i_target):
            # log('here4.2')
            sym_i_target.append(new)
            more_get_i = more_get_i - 1
            if more_get_i == 0:
                # log('here4.3')
                break

    sym_im = im
    i = 0
    for x in range(left_pix):
        if x < im.shape[1] / 2:
            x2 = im.shape[1] - (x + 1)
            for y in range(im.shape[0]):
                i = i + 1
                if not ismember(i, sym_i) and ismember(i, sym_i_target):
                    sym_im[y, x2] = im[y, x]

    return sym_im
