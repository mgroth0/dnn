from copy import deepcopy

import numpy as np
import scipy

from lib.misc import imutil
from lib.nn.nn_lib import RSA
from mlib.boot import log
from mlib.boot.lang import enum, islinux, listkeys
from mlib.boot.stream import arr, concat, flatten, isnan, listmap, randperm
from mlib.fig.makefigslib import MPLFigsBackend
from mlib.fig.PlotData import PlotData
from mlib.file import File, Folder, mkdir
from mlib.JsonSerializable import FigSet
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
        'log_by_mean'      : False
    },
    {
        'get_scores'       : False,
        'average_per_block': True,
        'log_by_mean'      : False
    },
    {
        'get_scores'       : False,
        'average_per_block': False,
        'log_by_mean'      : True,

    },
    {
        'get_scores'       : False,
        'average_per_block': True,
        'log_by_mean'      : True
    }
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
            log(f'net_folder:{net_folder}')
            modelname = net_folder.name
            if modelname not in activations:
                activations[modelname] = {}
            net_folder.delete_icon_file_if_exists()
            log(f'net_folder:{net_folder}: getting activations')
            the_files = net_folder.files
            for activations_mat in the_files.filtered(
                    lambda x: x.ext == 'mat'
            ):
                log(f'net_folder:{net_folder}: {activations_mat.name_pre_ext}')
                classname = activations_mat.name_pre_ext
                activations[modelname][classname] = activations_mat
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

                log('shortening')
                acts = listmap(shorten, acts)
                log('shortened!')

                if acts_for_rsa is None:
                    acts_for_rsa = acts
                else:
                    acts_for_rsa = concat(
                        acts_for_rsa,
                        acts,
                        axis=0
                    )

            fd = RSA(  # gets SIMILARITIES, not DiSSIMILARTIES due to fix()
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
                rsa_mat = fdd.data
                fdd.y_log_scale = cfg['log_by_mean']  # might not actually be by mean
                if cfg['average_per_block']:
                    for i, c in enum(CLASSES):
                        for ii, cc in enum(CLASSES):
                            if i < ii: continue
                            sc = slice(N_PER_CLASS * i, N_PER_CLASS * (i + 1))
                            sr = slice(N_PER_CLASS * ii, N_PER_CLASS * (ii + 1))
                            fdd.data[sc, sr] = np.mean(rsa_mat[sc, sr])

                full_data = fdd.data

                if not FORCED_RESOLUTION == fdd.data.shape[0]:
                    fdd.data = imutil.resampleim(np.array(fdd.data), FORCED_RESOLUTION, FORCED_RESOLUTION, nchan=1)[:,
                               :, 0]
                fdd.confuse_target = np.max(fdd.data)
                fdd.data = fdd.data.tolist()

                fdd.make = True
                extra = ''
                if cfg['average_per_block']: extra = '_avg'
                if cfg['log_by_mean']: extra += '_log'
                fdd.title = fdd.title + f'({extra})'
                file = result_folder[net + extra + ".mfig"]
                file.save(fdd)
                backend = MPLFigsBackend
                fdd = file.loado()
                fdd.dataFile = file
                fdd.imgFile = file.resrepext(IMAGE_FORMAT)
                backend.makeAllPlots([fdd], overwrite=True, force=False)
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
    breakpoint()
    debugData = {
        'scores'       : scores,
        'c_map'        : c_map,
        'result_folder': result_folder
    }
    File(f'temp{norm}.p').save(debugData)

NORMALIZE = True
norm = '_norm' if NORMALIZE else ''

@log_invokation
def debug_process(scores, result_folder, net, arch, size, plot, full_data):
    norm_rsa_mat = full_data / np.max(full_data)
    average = np.mean(norm_rsa_mat)

    similarity_NS = similarity_S = dissimilarity_across = similarity_across = 0
    NS_similarities = S_similarities = NS_to_S_similarities = []
    dvs = [0, 0, 0]
    debug = [[], [], []]
    for i, c in enum(CLASSES):
        for ii, cc in enum(CLASSES):
            if ii > i: continue
            comp_mat = norm_rsa_mat[
                slice(N_PER_CLASS * i, N_PER_CLASS * (i + 1)),
                slice(N_PER_CLASS * ii, N_PER_CLASS * (ii + 1))
            ]
            if c == cc:
                for cmi, row in enum(comp_mat):
                    for cmii, col in enum(row):
                        if cmi <= cmii:
                            comp_mat[cmi, cmii] = np.nan
            avg_dis = np.nanmean(comp_mat)
            all_dis = arr([num for num in flatten(comp_mat).tolist() if not isnan(num)])

            if NORMALIZE:
                avg_dis = avg_dis / average
                all_dis = all_dis / average

            if c.startswith('NS') and cc.startswith('NS'):
                similarity_NS += avg_dis
                dvs[0] += 1
                debug[0].append((c, cc))
                NS_similarities += flatten(all_dis).tolist()
            elif c.startswith('S') and cc.startswith('S'):
                similarity_S += avg_dis
                dvs[1] += 1
                debug[1].append((c, cc))
                S_similarities += flatten(all_dis).tolist()
            else:
                similarity_across += avg_dis
                dvs[2] += 1
                debug[2].append((c, cc))
                NS_to_S_similarities += flatten(all_dis).tolist()
                if NORMALIZE:
                    avg_dis = -avg_dis + 2
                else:
                    avg_dis = 1 - avg_dis
                dissimilarity_across += avg_dis

    similarity_NS = similarity_NS / dvs[0]
    similarity_S = similarity_S / dvs[1]
    dissimilarity_across = dissimilarity_across / dvs[2]
    similarity_across = similarity_across / dvs[2]

    NS_similarities = arr(NS_similarities)
    S_similarities = arr(S_similarities)
    NS_to_S_similarities = arr(NS_to_S_similarities)

    p_ns_s = scipy.stats.ttest_ind(NS_similarities, S_similarities, alternative='two-sided')[1]
    p_across_s = scipy.stats.ttest_ind(NS_to_S_similarities, S_similarities, alternative='less')[1]
    p_across_ns = scipy.stats.ttest_ind(NS_to_S_similarities, NS_similarities, alternative='less')[1]

    print(f'{p_ns_s=}')
    print(f'{p_across_s=}')
    print(f'{p_across_ns=}')

    result_folder[f"{net}_stats{norm}.json"].save({
        'p_ns_s'     : p_ns_s,
        'p_across_s' : p_across_s,
        'p_across_ns': p_across_ns
    })

    scores[arch][size] = {'AC': dissimilarity_across, 'S': similarity_S, 'NS': similarity_NS}[plot]

    VIOLIN = True

    y = [similarity_NS, similarity_S, similarity_across]
    if VIOLIN:
        y = [NS_similarities, S_similarities, NS_to_S_similarities]

    fd = PlotData(
        y=y,
        x=[
            'similarity_NS',
            'similarity_S',
            'similarity_across'
        ],
        item_type='violin' if VIOLIN else 'bar',
        item_color=[[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        ylim=[0, 20],
        title=f'{net}: Dissimilarities of {LAYERS[arch]}',
        err=() if VIOLIN else [np.std(NS_similarities), np.std(S_similarities), np.std(NS_to_S_similarities)],
        xlabel='Class Comparison Groups',
        ylabel='Dissimilarity Score',
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
