from matplotlib.lines import Line2D
import numpy as np

from lib.misc import imutil
from lib.nn.nn_lib import RSA, rsa_corr
from mlib.JsonSerializable import FigSet
from mlib.boot import log
from mlib.boot.lang import listkeys, enum, islinux
from mlib.boot.stream import concat, listmap, randperm, listitems
from mlib.fig.PlotData import PlotData
from mlib.fig.makefigslib import MPLFigsBackend
from mlib.file import Folder, mkdir, File

SANITY = False
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

# N_PER_CLASS = 10 #90 sec request, 324 total

# N_PER_CLASS = 80 #161 request, 1409 total
# N_PER_CLASS = 100  # 110 request, 2035 total
# alexnet is taking 1 or 2 seconds for 100 images , but gnet ~50 secs and IRN ~110 secs
# 100, after randperm and shorten each act to SQN len: 113 request, 464 total,
# total images per class: 500 always
# everything below uses shortening
# N_PER_CLASS = 200 #request 112, total 602

N_PER_CLASS = 500  # (max) # NO LONGER IN SLURM BC REQUEST WONT GO THROUGH, SO CPUS MIGHT SHARED WITH OTHER PROCESSES. NUM CPUS: 56, request intstant, 1915 total

# test
# N_PER_CLASS = 10

import multiprocessing
print(f'NUM CPUS: {multiprocessing.cpu_count()}')

LAYERS = {
    "SQN"      : 'relu_conv10',  # 784

    "AlexNet"  : 'fc7',  # 4096
    "GoogleNet": 'inception_5b-output',  # 50176
    "IRN"      : 'conv_7b_ac',  # 98304
    "IV3"      : 'mixed10',  # 131072
    "RN18"     : 'res5b-relu'  # 25088
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
    log('running rsa_for_darius')
    if islinux():
        imgActivations = Folder('/matt/data/imgActivationsForRSA')
    else:
        imgActivations = Folder('_data/imgActivationsForRSA')
    activations = {}

    for net_folder in imgActivations.files:
        log(f'net_folder:{net_folder}')
        if not net_folder.isdir:
            continue
        net_folder = Folder(net_folder)
        modelname = net_folder.name
        if modelname not in activations:
            activations[modelname] = {}
        arch, ntrain = modelname.split('_')
        # breakpoint()
        net_folder.delete_icon_file_if_exists()
        log(f'net_folder:{net_folder}: getting activations')
        print('b4 files')
        from mlib.boot import stream
        stream.enable_debug = True
        the_files = net_folder.files
        print('after files')
        for activations_mat in the_files.filtered(
                lambda x: x.ext == 'mat'
        ):
            log(f'net_folder:{net_folder}: {activations_mat.name_pre_ext}')
            classname = activations_mat.name_pre_ext
            activations[modelname][classname] = activations_mat

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
            # if arch != 'SQN': continue
            net = arch + '_' + str(size)
            # block_len = N_PER_CLASS
            block_len = 10

            acts_for_rsa = None

            for c in CLASSES:
                acts = activations[net][c].load()['imageActivations']
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

            # need to do this again after downsampling
            fd.confuse_target = np.max(fd.data)

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
            scores[arch][size] = dissimilarity_across
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

    c_map = {
        "SQN"      : [1, 0, 0],  # 784
        "AlexNet"  : [0, 0, 1],  # 4096
        "GoogleNet": [0, 1, 0],  # 50176
        "IRN"      : [1, 1, 0],  # 98304
        "IV3"      : [1, 0, 1],  # 131072
        "RN18"     : [0, 1, 1]  # 25088
    }
    fs = FigSet()

    debugData = {
        'scores'       : scores,
        'c_map'        : c_map,
        'result_folder': result_folder
    }
    File('temp.p').save(debugData)
    # breakpoint()



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


def test_line():
    fs = FigSet()
    debugData = File('temp.p').load()
    scores = debugData['scores']
    c_map = debugData['c_map']
    result_folder = debugData['result_folder']

    for akey, arch in listitems(scores):
        score_list = []
        size_list = []
        c_list = []
        for sizekey, score in listitems(arch):
            score_list.append(score)
            size_list.append(sizekey)
            c_list.append(c_map[akey])
        fd = PlotData(
            y=score_list,
            # xticklabels
            x=size_list,
            # item_type='scatter',
            item_type='line',
            # item_color=[[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            item_color=c_map[akey],  # ,c_list,
            ylim=[0, 20],
            # title=akey,
            # err=[0, 0, 0],
            xlabel='Training Sizes',
            ylabel=akey,#'Dissimilarity Score',
            # x=[1, 2, 3],
            bar_sideways_labels=False,
        )
        fd.make = True
        fs.viss.append(fd)
    fs.viss.append(PlotData(
        y=[],
        x=[],
        item_type='scatter',
        item_color=[],  # ,c_list,
        title='Dissimilarities',
        xlabel='Training Sizes',
        ylabel='Dissimilarity Score',
        bar_sideways_labels=False,
    ))
    fs.viss[-1].make = True
    fs.make = True
    file = result_folder["scatter.mfig"]
    file.save(fs)
    # backend = WolfMakeFigsBackend
    backend = MPLFigsBackend
    fs = file.loado()
    fs.file = file
    fs.imgFile = file.resrepext('png')
    # fs.viss[0].legend = listmap(
    #     # akey, arch
    #     lambda item: Line2D([0], [0], color=c_map[item[0]], lw=4, label=item[0]),
    #     listitems(scores)
    # )
    backend.makeAllPlots([fs], overwrite=True)


if __name__ == '__main__':
    if SANITY:
        sanity()
    else:
        main()
        # main2()
        test_line()
