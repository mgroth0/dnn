from copy import deepcopy

import numpy as np
from functools import lru_cache

from lib.datamodel.Mats import ComparisonMatrix, FeatureMatrix
from lib.datamodel.Statistical import StatisticalArrays
from lib.nn.nn_lib import OM_DATA_FOLDER
from lib.rsa_figs import method_strings, pattern_strings
from lib.RSA_sym_model import RSA_CLASSES, RSA_LAYERS
from mlib.boot.crunch import section
from mlib.boot.lang import enum, listkeys
from mlib.boot.mlog import progress
from mlib.boot.stream import arr, concat, flatten, isnan, randperm
from mlib.mat import rel_mat
from mlib.math import nan_above_eye, naneye
from mlib.str import suffix_int




@lru_cache()
def rsa_pattern(name, n_per_class, HIGH_IS_SIM=False):
    _pattern = SYM_CLASS_SET_PATTERNS[name]
    same_mat = rel_mat(lambda x, y: x == y, _pattern)
    if HIGH_IS_SIM:
        same_mat = np.where(same_mat, 1, 0)
    else:
        same_mat = np.where(same_mat, 0, 1)
    same_mat = same_mat.repeat(n_per_class, axis=0).repeat(n_per_class, axis=1)
    same_mat = naneye(same_mat.astype(float))
    same_mat = nan_above_eye(same_mat)
    return ComparisonMatrix(
        same_mat,
        method_used=f'PATTERN-{name}',
        class_set=RSA_CLASSES
    )









SYM_CLASS_SET_PATTERNS = {
    "sym"  : [1 if s.name.startswith('S') else 0 for s in RSA_CLASSES],
    "band" : [suffix_int(s.name) > 0 for s in RSA_CLASSES],
    "dark" : [1 if 'd' in s.name else 0 for s in RSA_CLASSES],
    "width": [int(suffix_int(s.name) / 2) for s in RSA_CLASSES]
}
SYM_CLASS_SET_PATTERN_GROUPINGS = {
    "sym"  : {'AC': [], 'NS': [], 'S': []},
    "band" : {'AC': [], 'NB': [], 'B': []},
    "dark" : {'AC': [], 'REG': [], 'D': []},
    "width": {'AC': [], '0': [], '2': [], '4': [], '6': []},
}

def separate_comp_mat_by_classes_compared(
        normalized1: ComparisonMatrix,
        net: str,
        arch: str,
        method_name: str,
        sim_string: str,
        pattern: str
) -> StatisticalArrays:
    simsets = deepcopy(SYM_CLASS_SET_PATTERN_GROUPINGS[pattern])
    simkeys = listkeys(SYM_CLASS_SET_PATTERN_GROUPINGS[pattern])[1:]
    for k in simsets:
        simsets[k] = arr()  # rather than list
    average = np.nanmean(normalized1)

    for i, c in enum(RSA_CLASSES):
        for ii, cc in enum(RSA_CLASSES):
            if ii > i: continue
            comp_mat = normalized1[c, cc]
            normalized2 = arr([num for num in flatten(comp_mat).tolist() if not isnan(num)]) / average
            groupname1 = simkeys[SYM_CLASS_SET_PATTERNS[pattern][i]]
            groupname2 = simkeys[SYM_CLASS_SET_PATTERNS[pattern][ii]]
            if groupname1 == groupname2:
                simsets[groupname1] += flatten(normalized2)
            else:
                simsets['AC'] += flatten(normalized2)

    return StatisticalArrays(
        ylabel=f'{sim_string} Score ({method_strings[method_name]}) ({pattern_strings[pattern]})',
        xlabel='Class Comparison Group',
        data={k: v for k, v in simsets.items()},
        title_suffix=f'{net}:{RSA_LAYERS[arch]} ({method_name}) ({pattern})'
    )

def darius_and_shobhita_acts(
        N_PER_CLASS,
        ACT_SIZE,
        INCLUDE_DARIUS,
        EXCLUDE_DARIUS_SMALLER_TRAIN_SIZES
):
    if INCLUDE_DARIUS:
        yield from _darius_and_shobhita_acts(
            N_PER_CLASS,
            ACT_SIZE,
            SHOBHITA=False,
            EXCLUDE_DARIUS_SMALLER_TRAIN_SIZES=EXCLUDE_DARIUS_SMALLER_TRAIN_SIZES
        )
    yield from _darius_and_shobhita_acts(
        N_PER_CLASS,
        ACT_SIZE,
        SHOBHITA=True,
        EXCLUDE_DARIUS_SMALLER_TRAIN_SIZES=EXCLUDE_DARIUS_SMALLER_TRAIN_SIZES
    )

def _darius_and_shobhita_acts(
        N_PER_CLASS,
        ACT_SIZE,
        SHOBHITA=True,
        EXCLUDE_DARIUS_SMALLER_TRAIN_SIZES=True
):
    T_SIZES = [
        25,
        50,
        100,
        150,
        200
    ]  # 6 epochs for all, and only 70% for training
    if EXCLUDE_DARIUS_SMALLER_TRAIN_SIZES:
        T_SIZES = [T_SIZES[-1]]
    NETS = listkeys(RSA_LAYERS)
    if SHOBHITA:
        # noinspection PyRedeclaration
        T_SIZES = [100000]  # num epochs?
        NETS = ["LSTM"]
    else:
        NETS.remove("LSTM")
    ACTIVATIONS_FOLDER = OM_DATA_FOLDER['activations']
    if not SHOBHITA:
        imgActivations = ACTIVATIONS_FOLDER.resolve('imgActivationsForRSA')
        activations = {}
        for net_folder in imgActivations.folders:
            modelname = net_folder.name
            if modelname not in activations:
                activations[modelname] = {}
            for activations_mat in net_folder.files.filtered(
                    lambda x: x.ext == 'mat'
            ):
                # breakpoint()
                activations[modelname][activations_mat.name_pre_ext] = activations_mat
    else:
        folder = ACTIVATIONS_FOLDER['rsa_activations_shobhita2']
        files = {f.name.split('Cat')[1].split('_')[0]: f for f in folder.files}
        activations = {
            'LSTM': {c.name: folder[files[c.name].name] for c in RSA_CLASSES}
        }
    for arch in NETS:
        arch_rand_perm = None
        for size in T_SIZES:
            net = arch
            with section(f'preparing {net} activations'):
                if not SHOBHITA: net = f'{net}_{size}'
                acts_for_rsa = None
                for c in [cls.name for cls in RSA_CLASSES]:
                    acts = activations[net][c].load(silent=True)
                    if not SHOBHITA:
                        acts = acts['imageActivations']


                    if arch_rand_perm is None:
                        progress(f"activation size of {net}: {len(acts[0])}")
                        arch_rand_perm = randperm(range(len(acts[0])))

                    acts = [a[arch_rand_perm][:ACT_SIZE] for a in acts[0:N_PER_CLASS]]

                    if acts_for_rsa is None:
                        acts_for_rsa = acts
                    else:
                        acts_for_rsa = concat(
                            acts_for_rsa,
                            acts,
                            axis=0
                        )
                yield arch, net, FeatureMatrix(
                    data=acts_for_rsa,
                    ground_truth=np.repeat(RSA_CLASSES, int(len(acts_for_rsa) / len(RSA_CLASSES))).tolist(),
                    class_set=RSA_CLASSES
                )
