import numpy as np

from lib.misc import imutil
from lib.nn.nn_lib import RSA, rsa_corr
from mlib.boot import log
from mlib.boot.lang import listkeys, enum, islinux
from mlib.boot.stream import concat
from mlib.fig.PlotData import PlotData
from mlib.fig.makefigslib import MPLFigsBackend
from mlib.file import Folder, mkdir, File

SANITY = True
SANITY_FILE = File('/Users/matt/Desktop/forMattActivs.mat')

# TRANSPOSE = True

# on openmind 10 took ~400 sec
# N_PER_CLASS = 10
# N_PER_CLASS = 100
# N_PER_CLASS = 15 # took 524 sec

# N_PER_CLASS = 20  # took 687, with multiprocess took 363 (and again, 362)! (56 CPUS)
# 100: 376, 347(2 sec slurm wait)
# 5(actually 80(twice?? must be wrong, since i put 5 and i think node has only 56??)!???): 578(4 sec slurm wait)
# 150(actually 56(twice)!???): 367(1 sec slurm wait)
# NONE(56)

# always requesting 60 (getting 56/80) from now on

# N_PER_CLASS = 40 #65 sec slurm request, said I have 80 cpus... took 533 total (468 after slurm)

N_PER_CLASS = 10

# N_PER_CLASS = 80

import multiprocessing
print(f'NUM CPUS: {multiprocessing.cpu_count()}')

LAYERS = {
    "AlexNet"  : 'fc7',
    "GoogleNet": 'inception_5b-output',
    "IRN"      : 'conv_7b_ac',
    "SQN"      : 'relu_conv10',
    "IV3"      : 'mixed10',
    "RN18"     : 'res5b-relu'
}
NETS = listkeys(LAYERS)
T_SIZES = [
    25,
    50,
    100,
    150,
    200
]
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




def main():
    if islinux():
        imgActivations = Folder('/matt/data/imgActivationsForRSA')
    else:
        imgActivations = Folder('_data/imgActivationsForRSA')
    activations = {}
    for net_folder in imgActivations.files:
        if not net_folder.isdir:
            continue
        net_folder = Folder(net_folder)
        modelname = net_folder.name
        if modelname not in activations:
            activations[modelname] = {}
        arch, ntrain = modelname.split('_')
        for activations_mat in net_folder.files.filtered(
                lambda x: x.ext == 'mat'
        ):
            classname = activations_mat.name_pre_ext
            activations[modelname][classname] = activations_mat

    result_folder = mkdir('_figs/rsa')

    for arch in NETS:
        for size in T_SIZES:
            # if arch != 'SQN': continue
            net = arch + '_' + str(size)
            # block_len = N_PER_CLASS
            block_len = 10

            acts_for_rsa = None

            for c in CLASSES:
                acts = activations[net][c].load()['imageActivations']
                acts = acts[0:N_PER_CLASS]

                if acts_for_rsa is None:
                    acts_for_rsa = acts
                else:
                    acts_for_rsa = concat(
                        acts_for_rsa,
                        acts,
                        axis=0
                    )
            fd = RSA(
                f'L2 Norm of {LAYERS[arch]} from {net}',
                acts_for_rsa,
                None,
                None,
                layer_name='fc7',
                layer_i=None,
                classnames=CLASSES,
                block_len=block_len,
                sort=False,
                return_result=True
            )

            log('resampling1')
            fd.data = imutil.resampleim(np.array(fd.data), len(CLASSES) * 10, len(CLASSES) * 10, nchan=1)[:, :,
                      0].tolist()
            log('resampled2')

            norm_rsa_mat = fd.data / np.max(fd.data)

            fd.make = True
            file = result_folder[net + ".mfig"]
            file.save(fd)
            backend = MPLFigsBackend
            fd = file.loado()
            fd.file = file
            fd.imgFile = file.resrepext('png')
            backend.makeAllPlots([fd], overwrite=True)

            dissimilarity_NS = 0
            dissimilarity_S = 0
            dissimilarity_across = 0

            for i, c in enum(CLASSES):
                # if i > 4: break
                for ii, cc in enum(CLASSES):
                    if ii < i: continue
                    sc = slice(block_len * i, block_len * (i + 1))
                    sr = slice(block_len * ii, block_len * (ii + 1))
                    comp_mat = norm_rsa_mat[sc, sr]
                    avg_dis = 1 - np.mean(comp_mat)

                    if c.startswith('NS') and cc.startswith('NS'):
                        dissimilarity_NS += avg_dis
                    elif c.startswith('S') and cc.startswith('S'):
                        dissimilarity_S += avg_dis
                    else:
                        dissimilarity_across += avg_dis

            fd = PlotData(
                y=[dissimilarity_NS, dissimilarity_S, dissimilarity_across],
                # xticklabels
                x=[
                    'dissimilarity_NS',
                    'dissimilarity_S',
                    'dissimilarity_across'
                ],
                item_type='bar',
                item_color=[[0, 0, 1], [0, 0, 1], [0, 0, 1]],
                ylim=[0, 20],
                title=f'Dissimilarities of {LAYERS[arch]} from {net}',
                err=[0, 0, 0],
                xlabel='Class Comparison Groups',
                ylabel='Dissimilarity Score',
                # x=[1, 2, 3],
                bar_sideways_labels=False
            )
            fd.make = True
            file = result_folder[net + "_dis.mfig"]
            file.save(fd)
            # backend = WolfMakeFigsBackend
            backend = MPLFigsBackend
            fd = file.loado()
            fd.file = file
            fd.imgFile = file.resrepext('png')
            backend.makeAllPlots([fd], overwrite=True)


def main2():
    data = []
    for arch in NETS:
        line = []
        for size in T_SIZES:
            dis = File(f'_figs/rsa/{arch}_{size}_dis.mfig').load()['viss'][0]['y'][2]
            line.append(dis)
        data.append(line)
    fd = PlotData(
        y=data,
        # xticklabels
        x=T_SIZES,
        item_type='line',
        item_color=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]],
        ylim=[0, 25],

        title=f'S-NS Dissimilarities Across Training Sizes',
        # err=[0, 0, 0],
        xlabel='Training Size',
        ylabel='Dissimilarity Score Between S and NS',
        # x=[1, 2, 3],
        # bar_sideways_labels=False,


        # callouts=NETS


    )
    fd.make = True
    result_folder = mkdir('_figs/rsa')
    file = result_folder["line.mfig"]
    file.save(fd)
    # backend = WolfMakeFigsBackend
    backend = MPLFigsBackend
    fd = file.loado()
    fd.file = file
    fd.imgFile = file.resrepext('png')
    backend.makeAllPlots([fd], overwrite=True)

def sanity():
    activations = SANITY_FILE.load()['activs']

    result_folder = mkdir('_figs/rsa')

    block_len = len(activations)

    fd = RSA(
        # f'L2 Norm of Sanity Test',
        f'Correlation Sanity Test, Transposed',
        activations.T,
        None,
        None,
        layer_name='fc7',
        layer_i=None,
        classnames=[''],
        block_len=block_len,
        sort=False,
        return_result=True,
        fun=rsa_corr
    )
    fd.make = True
    file = result_folder['sanity_transposed' + ".mfig"]
    file.save(fd)
    backend = MPLFigsBackend
    fd = file.loado()
    fd.file = file
    fd.imgFile = file.resrepext('png')
    backend.makeAllPlots([fd], overwrite=True)

if __name__ == '__main__':
    if SANITY:
        sanity()
    else:
        # main()
        main2()
