import glob
import random

from types import SimpleNamespace
import sys
from PIL import Image

import numpy as np

from lib.nn import nn_lib
from lib.misc.imutil import make1, make255, resampleim
import lib.nn.nnstate as nnstate
from mlib.boot import log
from mlib.boot.mlog import warn, err
from mlib.boot.stream import arr, randperm, concat, listitems
from mlib.err import assert_int
import mlib.file
from mlib.file import File
from mlib.math import iseven
from mlib.shell import shell
from mlib.term import log_invokation, Progress
LINUX_HOME = '/home/matt/'

class NN_Data_Dir(File):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = File(f'{self.abspath}_std')
    @log_invokation(with_args=True)
    def delete_norm_dir(self):
        return self.norm.deleteIfExists()

BLACK_AND_WHITE = False

class SymAsymClassPair:
    def __init__(self, bandsize, dark):
        self.bandsize = bandsize
        self.dark = dark

    def get_classnames(self):
        if self.dark:
            dark = 'd'
        else:
            dark = ''
        ns = f'NS{dark}{self.bandsize}'
        s = f'S{dark}{self.bandsize}'
        return ns, s

def get_class_dict(bands):
    classes = dict()
    n = 0
    for b in bands:
        ns, s = b.get_classnames()
        classes[ns] = n
        n = n + 1
        classes[s] = n
        n = n + 1
    return classes

@log_invokation()
def gen_images(*, folder, class_pairs, ims_per_class):
    N_IMAGES = ims_per_class * 2 * len(class_pairs)
    classes = get_class_dict(class_pairs)
    File(folder).deleteAllContents()

    BLOCK_HEIGHT_WIDTH = 20

    y = []

    band_group_size = N_IMAGES / len(class_pairs)
    band_group = 0
    band_group_i = 0

    with Progress(N_IMAGES) as prog:
        for i in range(N_IMAGES):
            im_data = np.random.rand(BLOCK_HEIGHT_WIDTH, BLOCK_HEIGHT_WIDTH)
            if BLACK_AND_WHITE:
                im_data = np.vectorize(round)(im_data)

            band = class_pairs[band_group]
            ns_classname, s_classname = band.get_classnames()
            darken = band.dark
            if darken:
                im_data = im_data / 2
            band = band.bandsize
            bar_start = int((BLOCK_HEIGHT_WIDTH / 2) - (band / 2))
            bar_end = bar_start + band
            for w in range(bar_start, bar_end):
                im_data[:, w] = 0.5

            band_group_i = band_group_i + 1
            if band_group_i == band_group_size:
                band_group = band_group + 1
                band_group_i = 0

            im_data = make255(im_data)

            if iseven(i):
                im_data = nn_lib.symm(im_data, 1)
                y.append(classes[s_classname])
                label = s_classname
            else:
                y.append(classes[ns_classname])
                label = ns_classname

            im_data = np.expand_dims(im_data, 2)

            # i think Darius' data was single channeled
            # im_data = np.concatenate((im_data, im_data, im_data), axis=2)

            im_file = File(f'{File(folder).abspath}/{label}/sym{i}.png')
            im_file.save(im_data, silent=True)
            prog.tick()

    return classes





GRAY_SCALE = True
SAVE_PREPROC_DATA = True




def getReal(
        image_HW,
        _class_label_map,
        normalize_single_ims,
        std_d,
        USING_STD_DIR
):
    warn('use preprocessor')
    real, HW = image_HW
    if GRAY_SCALE:
        real.data = Image.open(real.file.abspath)
    else:
        real.data = np.resize(
            arr(Image.open(real.file.abspath).getdata()),
            (20, 20, 3)
        )
    if normalize_single_ims and not USING_STD_DIR:
        # noinspection PyUnusedLocal
        def smallify():
            err('dev')
            files = glob.glob(sys.argv[1] + "/**/*.png", recursive=True)
            i = 0
            log('found ' + str(len(files)) + ' images')
            with Progress(len(files)) as prog:
                for f in files:
                    p = shell(['convert', f, '-resize', '20x20', f], silent=True)
                    p.interact()
                    i = i + 1
                    prog.tick()
            log('resized ' + str(i) + ' images')
            sys.exit()
        assert len(real.data.getdata()) == 20 * 20, 'dev: smallify if files are large but blocky'
        real.data = np.reshape(
            arr(
                real.data.getdata()
            ),
            (20, 20)
        )
        real.data = real.data / np.std(real.data)
        if SAVE_PREPROC_DATA:
            std_file = File(std_d).resolve(File(real.file).parentName).resolve(File(real.file).name)

            std_file.save(real.data, silent=True)

    real.data = make1(real.data)
    if normalize_single_ims:
        if GRAY_SCALE:
            real.data = real.data - np.mean(real.data)
        else:
            for chan in range(3):
                real.data[:, :, chan] = real.data[:, :, chan] - np.mean(real.data[:, :, chan])

    real.data = resampleim(real.data, HW, HW)
    if GRAY_SCALE:
        shape1 = real.data.shape[0]
        shape2 = real.data.shape[1]
        real.data = np.broadcast_to(real.data, (shape1, shape2, 3))
    real.label = _class_label_map[real.clazz]
    return real





# TRAIN_TEST_SPLIT = 0.7
def load_and_preprocess_ims(TRAIN_TEST_SPLIT, data_dir, normalize_single_images):
    norm_dir = data_dir.norm
    # global USING_STD_DIR
    USING_STD_DIR = False
    if norm_dir.exists and normalize_single_images:
        assert len(data_dir.glob('*/**/.png')) == len(norm_dir.glob('*/**/.png'))
        data_dir = norm_dir
        USING_STD_DIR = True

    log(f'running load_and_preprocess_ims for {data_dir}')

    if USING_STD_DIR:
        log('using std dir')
    elif normalize_single_images:
        log('running STD one time and saving to std dir')

    classnames = []
    labels = []
    class_label_map = {}
    next_label = 0

    for f in data_dir.files:
        if not f.isdir:
            log('problem with $', f)
            err('all files in data_dir should be folders')
        else:
            classnames.append(f.name)
            labels.append(next_label)
            class_label_map[f.name] = next_label
            next_label = next_label + 1
        for ff in f.files:
            if not ff.ext == 'png':
                log('problem with $', ff)
                err('all files in data_dir folders should be images')

    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if mlib.file.filename != "LICENSE.txt"])

    images = []
    for image in data_dir.glob('*/*.png'):
        images.append(image)

    _class_label_map = class_label_map

    log('loading $ images...', len(images))

    num_classes = len(data_dir.files)
    if num_classes == 0:
        numTrainPerClass = 0
    else:
        numTrainPerClass = round(TRAIN_TEST_SPLIT * (len(images) / num_classes))

    # RANDOMNESS HERE
    trainPerm = randperm(numTrainPerClass)

    trains = tests = dict()

    for nam in classnames:
        trains[nam] = []
        tests[nam] = []

    imagesR = []
    for im in images:
        real = SimpleNamespace()
        real.file = im
        real.clazz = real.file.parentName
        imagesR += [real]

    reals = dict()
    for r in CLASS_NAMES:
        reals[r] = []
    for im in imagesR:
        reals[im.clazz].append(im)

    for clazz, ims in reals.items():
        for idx, im in enumerate(ims):
            if idx in trainPerm:
                trains[clazz].append(im)
            else:
                tests[clazz].append(im)

    imdsTrain = arr(concat(*trains.values()))
    imdsValidation = arr(concat(*tests.values()))
    import random
    imdsTrain = imdsTrain.tolist()
    # RANDOMNESS HERE
    random.shuffle(imdsTrain)
    imdsTrain = arr(imdsTrain)

    imdsValidation = imdsValidation.tolist()
    # RANDOMNESS HERE
    random.shuffle(imdsValidation)
    imdsValidation = arr(imdsValidation)

    log('constructing PreDatasets')

    preT = PreDataset(imdsTrain, class_label_map,
                      normalize_single_images,
                      norm_dir,
                      USING_STD_DIR)
    preV = PreDataset(imdsValidation, class_label_map, normalize_single_images,
                      norm_dir,
                      USING_STD_DIR)

    log('returning PreDatasets')

    return preT, preV

class PreDataset:
    def __init__(self, imds, class_label_map, normalize_single_ims,
                 std_d, USING_STD_DIR):
        self.imds = imds
        self.class_label_map = class_label_map
        self.num_steps = assert_int(len(self.imds) / nnstate.FLAGS.batchsize)
        self.gen = None
        self.gen_for_ds = None

        self.normalize_single_ims = normalize_single_ims
        self.std_d = std_d
        self.USING_STD_DIR = USING_STD_DIR

    def __len__(self):
        return len(self.imds)

    def examples(self):
        examples = []
        if len(self.imds) > 0:
            for c, lab in listitems(self.class_label_map):
                break_outer = False
                for batch_pair in self.gen():
                    for img_pair in zip(*batch_pair):
                        if img_pair[1] == lab:
                            examples += [(c, img_pair[0])]
                            break_outer = True
                            break
                    if break_outer:
                        break

        return examples
    def prep(self, HW):
        # RANDOMNESS HERE
        random.shuffle(self.imds)

        # toRun = ziplist(imds, [HW for _ in imds])
        # imds = list(getReal(toRun))

        def gen():
            # twentyData = []
            # twentyLabel = []
            twentyPairs = []
            i = 0
            with Progress(len(self.imds)) as prog:
                for imd in self.imds:
                    i += 1
                    if i <= nnstate.FLAGS.batchsize:
                        twentyPairs += [getReal((imd, HW),
                                                self.class_label_map,
                                                self.normalize_single_ims,
                                                self.std_d,
                                                self.USING_STD_DIR)]
                        # twentyData.append(imd.data)
                        # twentyLabel.append(imd.label)
                    if i == nnstate.FLAGS.batchsize:
                        # batch = SimpleNamespace()
                        # batch.data = twentyData
                        # batch.label = twentyLabel
                        yield (
                            [imd.data for imd in twentyPairs],
                            [imd.label for imd in twentyPairs]
                        )
                        twentyPairs.clear()
                        # twentyData = []
                        # twentyLabel = []
                        i = 0

                    #     this is maybe better than logging in fill_cmat because it also works during net.predict()
                    prog.tick()


        self.gen = gen

        # why do I need this?
        # def gen_for_ds():
        #     yield from gen


        # for o in gen():
        #     yield o.data, o.label
        # self.gen_for_ds = gen_for_ds

        return self

    # def myseq(self):
    #     seq = MySeq(self.gen)
    #     return seq

    def xy(self):
        return [pair for pair in self.gen()]

    def x(self, net):
        x = list()
        for tup in self.xy():
            x.append(tup[0])
        ar = arr(x)
        shape = list(ar.shape[2:])
        shape.insert(0, -1)
        rrr = ar
        if net.ARCH_LABEL == 'INC' or net.ARCH_LABEL == 'SCRATCH':
            rrr = np.reshape(ar, tuple(shape))
        return rrr, shape

    def y(self, net):
        y = list()
        for tup in self.xy():
            y.append(arr(tup[1]).tolist())
        rrr = y

        if net.ARCH_LABEL == 'GNET' or net.ARCH_LABEL == 'INC' or net.ARCH_LABEL == 'SCRATCH':
            rrr = arr(y).flatten()
        return arr(rrr)

    def dataset(self, HEIGHT_WIDTH):
        import tensorflow as tf  # keep modular
        return tf.data.Dataset.from_generator(self.gen,
                                              (tf.float32, tf.int64),
                                              output_shapes=(
                                                  tf.TensorShape(
                                                      (nnstate.FLAGS.batchsize, HEIGHT_WIDTH, HEIGHT_WIDTH, 3)),
                                                  tf.TensorShape(([nnstate.FLAGS.batchsize]))))


# class MySeq(tensorflow.keras.utils.Sequence):
#     def __getitem__(self, index):
#         err('todo')
#     def __len__(self):
#         err('todo')
#     def __init__(self, gen):
#         self.gen = gen
#         xy = arr()
#         for g in gen():
#             xy += g
#         self.xy = xy
#
# def __len__(self):
#     return len(self.xy)
#
# def __getitem__(self, idx):
#     return self.xy[idx]
