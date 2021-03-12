from copy import deepcopy

import numpy as np
import scipy

from lib.misc import imutil
from lib.nn.nn_lib import RSA, rsa_corr
from mlib.boot import log
from mlib.boot.lang import enum, islinux, listkeys
from mlib.boot.mlog import err
from mlib.boot.stream import arr, concat, flatten, isnan, listitems, listmap, randperm
from mlib.fig.makefigslib import MPLFigsBackend
from mlib.fig.PlotData import PlotData
from mlib.file import File, Folder, mkdir
from mlib.JsonSerializable import FigSet
from mlib.term import log_invokation

SANITY = False
SANITY_FILE = File('/Users/matt/Desktop/forMattActivs.mat')

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



DEBUG = True

def main():
    log('running rsa_for_darius')
    if not SHOBHITA:
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
            # stream.enable_debug = True
            the_files = net_folder.files
            print('after files')
            for activations_mat in the_files.filtered(
                    lambda x: x.ext == 'mat'
            ):
                log(f'net_folder:{net_folder}: {activations_mat.name_pre_ext}')
                classname = activations_mat.name_pre_ext
                activations[modelname][classname] = activations_mat
    else:
        folder = Folder('/matt/data/rsa_activations_shobhita2')
        activations = {'LSTM': {}}
        files = {f.name.split('Cat')[1].split('_')[0]: f for f in folder.files}
        for c in CLASSES:
            activations['LSTM'][c] = folder[files[c].name]

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
            if SHOBHITA:
                net = arch
            else:
                net = arch + '_' + str(size)

            # block_len = 100  # PIXELS_PER_CLASS
            # block_len = 500  # PIXELS_PER_CLASS



            # block_len = 100 if SHOBHITA else 10
            block_len = 10 #DEBUG

            acts_for_rsa = None

            for c in CLASSES:
                if SHOBHITA:
                    # 500 = num images
                    # 400 = len of one activation array
                    # breakpoint()
                    # acts = activations[net][c].load().reshape(500, 400)
                    breakpoint()
                    if DEBUG:
                        pass
                    acts = activations[net][c].load()
                else:
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

            # breakpoint()
            fd = RSA(  # gets SIMILARITIES, not DiSSIMILARTIES due to fix()
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
            for cfg in [
                {
                    'get_scores'       : True,
                    'average_per_block': False,
                    'log_by_mean': False
                },
                {
                    'get_scores'       : False,
                    'average_per_block': True,
                    'log_by_mean': False
                },
                {
                    'get_scores'       : False,
                    'average_per_block': False,
                    'log_by_mean': True,

                },
                {
                    'get_scores'       : False,
                    'average_per_block': True,
                    'log_by_mean': True
                }
            ]:
                fdd = deepcopy(fd)
                rsa_mat = fdd.data
                if cfg['average_per_block']:
                    for i, c in enum(CLASSES):
                        for ii, cc in enum(CLASSES):
                            if ii < i: continue
                            sc = slice(N_PER_CLASS * i, N_PER_CLASS * (i + 1))
                            sr = slice(N_PER_CLASS * ii, N_PER_CLASS * (ii + 1))
                            comp_mat = rsa_mat[sc, sr]
                            avg_dis = np.mean(comp_mat)
                            # fd.data = arr(fd.data)
                            breakpoint()
                            fdd.data[sc, sr] = avg_dis
                            # fd.data = fd.data.tolist()

                # breakpoint()

                log('resampling1')
                lennnn = len(CLASSES) * block_len
                full_data = fdd.data
                if lennnn == fdd.data.shape[0]:
                    fdd.data = fdd.data.tolist()
                else:
                    # DEBUG
                    # for rowi,row in enum(fd.data):
                    #     copy = fd.data[rowi]
                    #     random.shuffle(copy)
                    #     fd.data[rowi] = copy



                    fdd.data = imutil.resampleim(np.array(fdd.data), lennnn, lennnn, nchan=1)[:, :, 0].tolist()
                log('resampled2')

                # need to do this again after downsampling
                fdd.confuse_target = np.max(fdd.data)

                fdd.make = True
                extra = ''
                if cfg['average_per_block']: extra = '_avg'
                if cfg['log_by_mean']: extra += '_log'
                file = result_folder[net + extra + ".mfig"]
                file.save(fdd)
                backend = MPLFigsBackend
                fdd = file.loado()
                fdd.dataFile = file
                fdd.imgFile = file.resrepext('png')
                backend.makeAllPlots([fdd], overwrite=True)
                if cfg['get_scores']:
                    scores = debug_process(fdd, scores, result_folder, net, block_len, arch, size, 'AC',full_data)

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
    fs = FigSet()

    debugData = {
        'scores'       : scores,
        'c_map'        : c_map,
        'result_folder': result_folder
    }
    File(f'temp{norm}.p').save(debugData)

@log_invokation
def debug_process_post(plot):
    result_folder = mkdir('_figs/rsa')
    sqn_act_len = None

    scores = {}
    for arch in NETS:
        log(f'in arch: {arch}')
        scores[arch] = {}
        arch_rand_perm = None
        for size in T_SIZES:
            if SHOBHITA:
                net = arch
            else:
                net = arch + '_' + str(size)
            block_len = 10

            acts_for_rsa = None

            file = result_folder[net + ".mfig"]
            fd = file.loado()
            fd.dataFile = file
            fd.imgFile = file.resrepext('png')

            scores = debug_process(fd, scores, result_folder, net, block_len, arch, size, plot)
    save_scores(result_folder, scores)


NORMALIZE = True
norm = ''
if NORMALIZE: norm = '_norm'

@log_invokation
def debug_process(fd, scores, result_folder, net, block_len, arch, size, plot,full_data):
    # fd = fd.viss[0]  # so confused why i have to do this locally but not on OM

    norm_rsa_mat = full_data / np.max(full_data)
    average = np.mean(norm_rsa_mat)

    similarity_NS = 0
    similarity_S = 0
    dissimilarity_across = 0
    similarity_across = 0

    similarity_NS_flat = []
    similarity_S_flat = []
    sim_across_flat = []

    # total_n = len(CLASSES) * len(CLASSES)
    dvs = [0, 0, 0]

    debug = [[], [], []]

    for i, c in enum(CLASSES):
        # if i > 4: break
        for ii, cc in enum(CLASSES):
            if ii < i: continue

            # sc = slice(block_len * i, block_len * (i + 1))
            # sr = slice(block_len * ii, block_len * (ii + 1))
            #
            sc = slice(N_PER_CLASS * i, N_PER_CLASS * (i + 1))
            sr = slice(N_PER_CLASS * ii, N_PER_CLASS * (ii + 1))

            comp_mat = norm_rsa_mat[sc, sr]
            if c == cc:
                for cmi, row in enum(comp_mat):
                    for cmii, col in enum(row):
                        if cmi <= cmii:
                            comp_mat[cmi, cmii] = np.nan
            avg_dis = np.nanmean(comp_mat)
            all_dis = arr([num for num in flatten(comp_mat).tolist() if not isnan(num)])

            # len([n for n in flatten(comp_mat).tolist() if n > 100])


            if NORMALIZE:
                avg_dis = avg_dis / average
                all_dis = all_dis / average

            if c.startswith('NS') and cc.startswith('NS'):
                similarity_NS += avg_dis
                dvs[0] += 1
                debug[0].append((c, cc))
                similarity_NS_flat += flatten(all_dis).tolist()
            elif c.startswith('S') and cc.startswith('S'):
                similarity_S += avg_dis
                dvs[1] += 1
                debug[1].append((c, cc))
                similarity_S_flat += flatten(all_dis).tolist()
            else:
                similarity_across += avg_dis
                dvs[2] += 1
                debug[2].append((c, cc))
                sim_across_flat += flatten(all_dis).tolist()
                if NORMALIZE:
                    # avg_dis = avg_dis - ((avg_dis - 1) * 2)
                    # avg_dis = avg_dis - ((2 * avg_dis) - 2)
                    # avg_dis = avg_dis - (2 * avg_dis) + 2
                    # avg_dis = (-2 * avg_dis) + 2 + avg_dis
                    avg_dis = -avg_dis + 2
                else:
                    avg_dis = 1 - avg_dis
                dissimilarity_across += avg_dis

    similarity_NS = similarity_NS / dvs[0]
    similarity_S = similarity_S / dvs[1]
    dissimilarity_across = dissimilarity_across / dvs[2]
    similarity_across = similarity_across / dvs[2]

    similarity_NS_flat = arr(similarity_NS_flat)
    similarity_S_flat = arr(similarity_S_flat)
    sim_across_flat = arr(sim_across_flat)


    p_ns_s = scipy.stats.ttest_ind(similarity_NS_flat, similarity_S_flat, alternative='two-sided')[1]
    p_across_s = scipy.stats.ttest_ind(sim_across_flat, similarity_S_flat, alternative='less')[1]
    p_across_ns = scipy.stats.ttest_ind(sim_across_flat, similarity_NS_flat, alternative='less')[1]

    # scipy.stats.ttest_ind(similarity_NS_flat, similarity_S_flat, alternative='two-sided')
    # scipy.stats.ttest_ind(sim_across_flat, similarity_S_flat, alternative='less')

    print(f'{p_ns_s=}')
    print(f'{p_across_s=}')
    print(f'{p_across_ns=}')

    result_folder[net + f"_stats{norm}.json"].save({
        'p_ns_s'     : p_ns_s,
        'p_across_s' : p_across_s,
        'p_across_ns': p_across_ns
    })

    # try:

    similarity_NS_std = np.std(similarity_NS_flat)
    similarity_S_stf = np.std(similarity_S_flat)
    dissimilarity_across_std = np.std(sim_across_flat)
    # except:
    #     breakpoint()

    if plot == 'AC':
        scores[arch][size] = dissimilarity_across
    elif plot == 'S':
        scores[arch][size] = similarity_S
    elif plot == 'NS':
        scores[arch][size] = similarity_NS
    else:
        err('bad')

    fd = PlotData(
        y=[similarity_NS, similarity_S, similarity_across],
        x=[
            'similarity_NS',
            'similarity_S',
            'similarity_across'
        ],
        item_type='bar',
        item_color=[[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        ylim=[0, 20],
        title=f'{net}: Dissimilarities of {LAYERS[arch]}',
        err=[similarity_NS_std, similarity_S_stf, dissimilarity_across_std],
        xlabel='Class Comparison Groups',
        ylabel='Dissimilarity Score',
        bar_sideways_labels=False
    )
    fd.make = True
    fd.title_size = 20
    file = result_folder[net + f"_dis{norm}.mfig"]
    fs = FigSet()
    fs.viss.append(fd)
    file.save(fs)
    backend = MPLFigsBackend
    fd = file.loado()
    fd.dataFile = file
    fd.imgFile = file.resrepext('png')

    backend.makeAllPlots([fd], overwrite=True)
    return scores




def main2():
    data = []
    for arch in NETS:
        line = []
        for size in T_SIZES:
            dis = File(f'_figs/rsa/{arch}_{size}_dis{norm}.mfig').load()['viss'][0]['y'][2]
            line.append(dis)
        data.append(line)
    fd = PlotData(
        y=data,
        x=T_SIZES,
        item_type='line',
        item_color=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]],
        ylim=[0, 25],

        title=f'S-NS Dissimilarities Across Training Sizes',
        xlabel='Training Size',
        ylabel='Dissimilarity Score Between S and NS',
    )

    fd.make = True
    result_folder = mkdir('_figs/rsa')
    file = result_folder[f"line{norm}.mfig"]
    file.save(fd)
    backend = MPLFigsBackend
    fd = file.loado()
    fd.dataFile = file
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
    fd.dataFile = file
    fd.imgFile = file.resrepext('png')
    backend.makeAllPlots([fd], overwrite=True)


PLOTS = {
    'S' : 'Symmetry Representation Homogeneity',
    'NS': 'Asymmetry Representation Homogeneity',
    'AC': 'Dissimilarity Between Symmetry and Asymmetry Representations'
}

def test_line(plot):
    fs = FigSet()
    debugData = File(f'temp{norm}.p').load()
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
            x=size_list,
            # item_type='scatter',
            item_type='line',
            # item_color=[[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            item_color=c_map[akey],  # ,c_list,
            ylim=[0, 20],
            # title=akey,
            # err=[0, 0, 0],
            xlabel='Training Sizes',
            ylabel=akey,  # 'Dissimilarity Score',
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
        title=PLOTS[plot],
        xlabel='Training Sizes',
        ylabel='Dissimilarity Score' if plot == 'AC' else 'Homogeneity Score',
        bar_sideways_labels=False,
    ))
    fs.viss[-1].make = True
    fs.make = True
    file = result_folder[f"scatter_{plot}{norm}.mfig"]
    for vis in fs.viss:
        vis.title_size = 25
    file.save(fs)
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
        test_line('AC')
