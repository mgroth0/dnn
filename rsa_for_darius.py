from copy import deepcopy

import numpy as np
from functools import lru_cache

from lib.misc import imutil
from lib.nn.nn_lib import RSA_GETS_SIMS_NOT_DESIMS
from mlib.boot import log
from mlib.boot.lang import enum, islinux, listkeys, listvalues
from mlib.boot.mlog import err
from mlib.boot.stream import arr, concat, flatten, isnan, itr, listmap, randperm
from mlib.fig.makefigslib import MPLFigsBackend
from mlib.fig.PlotData import PlotData
from mlib.file import File, Folder, mkdir
from mlib.JsonSerializable import FigSet
from mlib.stats import ttests
from mlib.term import log_invokation

N_PER_CLASS = 5
# N_PER_CLASS = 500

# BLOCK_LEN = 100 if SHOBHITA else 10
# BLOCK_LEN = 10  # DEBUG
# if block_len != n_per_class, resampling happens
BLOCK_LEN = N_PER_CLASS
DEBUG_DOWNSAMPLE = slice(0, None, 100)
CFG = [
          {
              'get_scores'       : True,
              'average_per_block': False,
              'log_by_mean'      : False,
              'pattern'          : None
          },
          {
              'get_scores'       : False,
              'average_per_block': True,
              'log_by_mean'      : False,
              'pattern'          : None
          },
          {
              'get_scores'       : False,
              'average_per_block': False,
              'log_by_mean'      : True,
              'pattern'          : None
          },
          {
              'get_scores'       : False,
              'average_per_block': True,
              'log_by_mean'      : True,
              'pattern'          : None
          }
      ] + [
          {
              'get_scores'       : False,
              'average_per_block': False,
              'log_by_mean'      : False,
              'pattern'          : name
          } for name in ["sym", "band", "dark", "width"]
      ]

# IMAGE_FORMAT = 'png'
IMAGE_FORMAT = 'svg'

# (max) # NO LONGER IN SLURM BC REQUEST WONT GO THROUGH, SO CPUS MIGHT SHARED WITH OTHER PROCESSES. NUM CPUS: 56, request intstant, 1915 total

# test
# N_PER_CLASS = 10

SHOBHITA = True

LAYERS = {
    "SQN"      : 'relu_conv10',  # 784
    "AlexNet"  : 'fc7',  # 4096
    "GoogleNet": 'inception_5b-output',  # 50176
    "IRN"      : 'conv_7b_ac',  # 98304
    "IV3"      : 'mixed10',  # 131072
    "RN18"     : 'res5b-relu',  # 25088,
    "LSTM"     : 'final cell'
}
if SHOBHITA:
    NETS = ["LSTM"]
else:
    NETS = listkeys(LAYERS)
T_SIZES = [
    25,
    50,
    100,
    150,
    200
]  # 6 epochs for all, and only 70% for training
if SHOBHITA:
    T_SIZES = [100000]  # num epochs?
CLASSES = [
    'NS0',
    'NS2',
    'NS4',
    'NS6',
    'NSd4',
    'S0',
    'S2',
    'S4',
    'S6',
    'Sd4'
]

FORCED_RESOLUTION = len(CLASSES) * BLOCK_LEN

SINGULARITY_DATA_FOLDER = Folder('/matt/data')
# OM_DATA_FOLDER = SINGULARITY_DATA_FOLDER
OM_DATA_FOLDER = Folder('/om2/user/mjgroth/data')

MAC_DATA_FOLDER = Folder('_data')

DATA_FOLDER = OM_DATA_FOLDER if islinux() else MAC_DATA_FOLDER



def main():
    if not SHOBHITA:
        imgActivations = DATA_FOLDER.resolve('imgActivationsForRSA')
        activations = {}
        for net_folder in imgActivations.folders:
            modelname = net_folder.name
            if modelname not in activations:
                activations[modelname] = {}
            for activations_mat in net_folder.files.filtered(
                    lambda x: x.ext == 'mat'
            ):
                activations[modelname][activations_mat.name_pre_ext] = activations_mat
    else:
        folder = DATA_FOLDER['rsa_activations_shobhita2']
        files = {f.name.split('Cat')[1].split('_')[0]: f for f in folder.files}
        activations = {
            'LSTM': {c: folder[files[c].name] for c in CLASSES}
        }

    log(f'finished net_folder loop')
    result_folder = mkdir('_figs/rsa')
    result_folder.clear()
    sqn_act_len = None

    scores = {}
    for arch in NETS:
        log(f'in arch: {arch}')
        scores[arch] = {}
        arch_rand_perm = None
        for size in T_SIZES:
            net = arch
            if not SHOBHITA: net = f'{net}_{size}'
            acts_for_rsa = None
            for c in CLASSES:
                acts = activations[net][c]
                if not SHOBHITA:
                    acts = acts['imageActivations']
                acts = acts[DEBUG_DOWNSAMPLE]
                log(f'total images per class: {len(acts)}')
                log(f'total acts per image: {len(acts[0])}')
                if sqn_act_len is None:
                    sqn_act_len = len(acts[0])

                if arch_rand_perm is None:
                    arch_rand_perm = randperm(range(len(acts[0])))

                acts = acts[0:N_PER_CLASS]

                only_one = {1: True}
                def shorten(a):
                    if only_one[1]:
                        log(f'shortening {len(a)} to {len(arch_rand_perm)} to {sqn_act_len}')
                        only_one[1] = False
                    return a[arch_rand_perm][0:sqn_act_len]

                acts = listmap(shorten, acts)

                if acts_for_rsa is None:
                    acts_for_rsa = acts
                else:
                    acts_for_rsa = concat(
                        acts_for_rsa,
                        acts,
                        axis=0
                    )

            fd = RSA_GETS_SIMS_NOT_DESIMS(
                f'L2 Norm of {LAYERS[arch]} from {net}',
                acts_for_rsa,
                None,
                None,
                layer_name='fc7',
                layer_i=None,
                classnames=CLASSES,
                block_len=BLOCK_LEN,
                sort=False,
                return_result=True,

            )
            for cfg in CFG:
                fdd = deepcopy(fd)
                pattern = cfg['pattern']
                if pattern:
                    fdd.data = _pattern(pattern)
                fdd.y_log_scale = cfg['log_by_mean']  # might not actually be by mean
                if cfg['average_per_block']:
                    for i, c in enum(CLASSES):
                        for ii, cc in enum(CLASSES):
                            if i < ii: continue
                            sc = slice(N_PER_CLASS * i, N_PER_CLASS * (i + 1))
                            sr = slice(N_PER_CLASS * ii, N_PER_CLASS * (ii + 1))
                            fdd.data[sc, sr] = np.mean(fdd.data[sc, sr])

                full_data = fdd.data

                if not FORCED_RESOLUTION == fdd.data.shape[0]:
                    fdd.data = imutil.resampleim(
                        np.array(fdd.data),
                        FORCED_RESOLUTION,
                        FORCED_RESOLUTION,
                        nchan=1
                    )[:, :, 0]
                fdd.confuse_target = np.max(fdd.data)
                fdd.data = fdd.data.tolist()
                # breakpoint()
                fdd.make = True
                extra = ''
                if cfg['average_per_block']: extra = '_avg'
                if cfg['log_by_mean']: extra += '_log'
                if pattern:
                    fdd.title = pattern
                    file = result_folder[f"{pattern}.mfig"]
                else:
                    fdd.title = f'{fdd.title}({extra})'
                    file = result_folder[f"{net}{extra}.mfig"]
                file.save(fdd)
                backend = MPLFigsBackend
                fdd = file.loado()
                fdd.dataFile = file
                fdd.imgFile = file.resrepext(IMAGE_FORMAT)
                backend.makeAllPlots([fdd], overwrite=True, force=False,
                                     # debug=bool(pattern)
                                     )
                if cfg['get_scores']:
                    scores = debug_process(scores, result_folder, net, arch, size, 'AC', full_data)

    save_scores(result_folder, scores)
def save_scores(result_folder, scores):
    c_map = {
        "SQN"      : [1, 0, 0],  # 784
        "AlexNet"  : [0, 0, 1],  # 4096
        "GoogleNet": [0, 1, 0],  # 50176
        "IRN"      : [1, 1, 0],  # 98304
        "IV3"      : [1, 0, 1],  # 131072
        "RN18"     : [0, 1, 1],  # 25088,
        "LSTM"     : [0, 0, 0]
    }
    File(f'temp{norm}.p').save(locals())

NORMALIZE = True
norm = '_norm' if NORMALIZE else ''

@log_invokation
def debug_process(scores, result_folder, net, arch, size, plot, full_data):
    norm_rsa_mat = full_data / np.max(full_data)
    average = np.mean(norm_rsa_mat)
    simsets = {'AC': [], 'S': [], 'NS': []}
    for i, c in enum(CLASSES):
        for ii, cc in enum(CLASSES):
            if ii > i: continue
            comp_mat = norm_rsa_mat[
                slice(N_PER_CLASS * i, N_PER_CLASS * (i + 1)),
                slice(N_PER_CLASS * ii, N_PER_CLASS * (ii + 1))
            ]
            if c == cc:
                for cmi in itr(comp_mat):
                    comp_mat[cmi, slice(cmi, None)] = np.nan
            all_dis = arr([num for num in flatten(comp_mat).tolist() if not isnan(num)])

            if NORMALIZE:
                all_dis = all_dis / average

            if c.startswith('NS') and cc.startswith('NS'):
                si = 'NS'
            elif c.startswith('S') and cc.startswith('S'):
                si = 'S'
            else:
                si = 'AC'
            simsets[si] += flatten(all_dis).tolist()
    simsets = {k: arr(v) for k, v in simsets.items()}
    result_folder[f"{net}_stats{norm}.json"].save(ttests(simsets))
    means = {k: np.nanmean(v) for k, v in simsets.items()}
    scores[arch][size] = means[plot]
    VIOLIN = True
    y = listvalues(simsets) if VIOLIN else listvalues(means)
    fd = PlotData(
        y=y,
        x=listkeys(simsets),
        item_type='violin' if VIOLIN else 'bar',
        item_color=[[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        ylim=[0, 20],
        title=f'{net}: Similarity scores of {LAYERS[arch]}',
        err=() if VIOLIN else [np.std(v) for v in simsets.values()],
        xlabel='Class Comparison Groups',
        ylabel='Similarity Score',
        bar_sideways_labels=False
    )
    fd.make = True
    fd.title_size = 20
    file = result_folder[f"{net}_dis{norm}.mfig"]
    fs = FigSet()
    fs.viss.append(fd)
    file.save(fs)
    fd = file.loado()
    fd.dataFile = file
    fd.imgFile = file.resrepext(IMAGE_FORMAT)
    MPLFigsBackend.makeAllPlots([fd], overwrite=True)


@lru_cache()
def _pattern(name, n_per_class=N_PER_CLASS):
    length = n_per_class * 10
    half = int(length / 2)
    mat = np.zeros((length, length))
    if name == 'sym':
        mat[:half, :half] = 1
        mat[half:, half:] = 1
    elif name == 'band':
        mat[:n_per_class, :n_per_class] = 1
        mat[n_per_class:half, n_per_class:half] = 1
        example = mat[:half, :half]
        mat[half:, :half] = example
        mat[:half, half:] = example
        mat[half:, half:] = example
    elif name == 'dark':
        v4 = half - n_per_class
        mat[:v4, :v4] = 1
        mat[v4:half, v4:half] = 1
        example = mat[:half, :half]
        mat[half:, :half] = example
        mat[:half, half:] = example
        mat[half:, half:] = example
    elif name == 'width':
        for n in range(5):
            s = slice(n * n_per_class, (n + 1) * n_per_class)
            mat[s, s] = 1
        mat[4 * n_per_class:5 * n_per_class, 2 * n_per_class:3 * n_per_class] = 1
        mat[2 * n_per_class:3 * n_per_class, 4 * n_per_class:5 * n_per_class] = 1
        example = mat[:half, :half]
        mat[half:, :half] = example
        mat[:half, half:] = example
        mat[half:, half:] = example
    else:
        err(f'unknown pattern: {name}')
    # breakpoint()
    return mat
