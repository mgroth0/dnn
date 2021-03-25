from itertools import chain

from lib.boot import nn_init_fun
from lib.salience.filter.salience_filter import MattSalienceFilter
from mlib.boot.lang import enum
from mlib.str import utf_decode
from rsa_for_darius import DATA_FOLDER
print('gen_preproc_ims.py: top')
import glob
import random

from types import SimpleNamespace
import sys
from PIL import Image

import numpy as np

from lib.nn import nn_lib
from lib.misc.imutil import make1, make255, resampleim
import lib.nn.nnstate as nnstate
from lib.preprocessor import preprocessors
from mlib.boot import log
from mlib.boot.mlog import warn, err
from mlib.boot.stream import arr, ints, randperm, concat, listitems, unique
from mlib.err import assert_int
import mlib.file
from mlib.file import File
from mlib.math import iseven
from mlib.shell import shell
from mlib.term import log_invokation, Progress
print('gen_preproc_ims.py: finished imports')
LINUX_HOME = '/home/matt/'



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

            # im_file = File(f'{File(folder).abspath}/{label}/sym{i}.png')
            im_file = File(f'{File(folder).abspath}/{label}/{label}_{band_group_i}.png')

            band_group_i = band_group_i + 1
            if band_group_i == band_group_size:
                band_group = band_group + 1
                band_group_i = 0

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
            the_name = f.name
            if nnstate.use_reduced_map and the_name in nnstate.reduced_map:
                the_name = nnstate.reduced_map[the_name]
            if the_name not in classnames:
                classnames.append(the_name)
                labels.append(next_label)
                class_label_map[the_name] = next_label
                next_label = next_label + 1
        for ff in f.files:
            if not ff.ext == 'png' and not ff.abspath.split('.')[-1] in ['png', 'jpg', 'jpeg']:  # cat.123.png
                log('problem with $', ff)
                err('all files in data_dir folders should be images')

    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if mlib.file.filename != "LICENSE.txt"])
    if nnstate.use_reduced_map:
        CLASS_NAMES = unique(list(nnstate.reduced_map.values()))
        # if the_name in nnstate.reduced_map:
        #     the_name = nnstate.reduced_map[the_name]

    log(f'CLASS_NAMES:{CLASS_NAMES}')

    images = []
    for image in data_dir.glob('*/*.png'):
        images.append(image)
    for image in data_dir.glob('*/*.jpg'):
        images.append(image)
    for image in data_dir.glob('*/*.jpeg'):
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
        the_name = real.file.parentName
        if nnstate.use_reduced_map and the_name in nnstate.reduced_map:
            the_name = nnstate.reduced_map[the_name]
        real.clazz = the_name
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

        # breakpoint()
        return examples
    # File('/matt/data/temp.png').save(File('/matt/data/tf_bug1/dogscats/ntrain/dog/dog.1648.jpg').load())
    # File(imd.file))
    # File('/matt/data/temp.png').save(preprocessors(HW)[pp_type].preprocess(File('/matt/data/tf_bug1/dogscats/ntrain/dog/dog.1648.jpg')))
    def prep(self, HW, pp_type):
        # RANDOMNESS HERE
        random.shuffle(self.imds)

        # toRun = ziplist(imds, [HW for _ in imds])
        # imds = list(getReal(toRun))

        def gen():
            # twentyData = []
            # twentyLabel = []
            twentyPairs = []
            i = 0

            # did this?
            warn('NEED TO MERGE getReal and PREPROCESSOR CODE. USE PREPROCESSOR.')
            sfilt = MattSalienceFilter()
            with Progress(len(self.imds)) as prog:
                for imd in self.imds:
                    i += 1
                    if i <= nnstate.FLAGS.batchsize:
                        if nnstate.FLAGS.salience:
                            the_new = imd

                            data = File(imd.file).load()
                            if nnstate.FLAGS.cfg_cfg['full_cfg']['SFILT']:
                                data = sfilt.experiment_function_pre_preprocess(data)

                            the_new.data = preprocessors(HW)[pp_type].preprocess(data)

                            # I think I fixed this. problem was preprocess resize was not resizing if one of the dimensions was right but not the other. Used an 'and' when I should have used an 'or'.
                            # if (str(type(the_new.data)) != "<class 'numpy.ndarray'>") or (
                            #         str(the_new.data.dtype) != "float32") or str(
                            #         the_new.data.shape) != '(299, 299, 3)':  # debug
                            #     breakpoint()
                            # log('finished preprocess')
                            the_new.label = self.class_label_map[imd.clazz]
                        else:
                            the_new = getReal((imd, HW),
                                              self.class_label_map,
                                              self.normalize_single_ims,
                                              self.std_d,
                                              self.USING_STD_DIR)

                        twentyPairs += [
                            the_new


                        ]
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

@log_invokation()
def gen_main(FLAGS, _IMAGES_FOLDER, HUMAN_IMAGE_FOLDER):
    log('in gen!')
    _IMAGES_FOLDER.clearIfExists()
    HUMAN_IMAGE_FOLDER.clearIfExists()
    gen_cfg = FLAGS.cfg_cfg['gen_cfg']

    #  these numbers might be lower now that I'm excluding images that aren't squares
    # nevermind. I think trying to only take squares didn't work
    cats = ['Egyptian cat',  # >=200
            'Siamese cat',  # 196
            'Persian cat',  # >=200
            'tiger cat',  # 182
            'tabby cat']  # >=100
    dogs = [
        'Afghan hound',  # >=200
        'basset hound',  # >=200
        'beagle',  # 198
        'bloodhound',  # 199
        'bluetick'  # >=100
    ]
    classes = cats + dogs
    not_trained = ['tabby cat', 'bluetick']
    for d in dogs:
        nnstate.reduced_map[d] = 'dog'
    for c in cats:
        nnstate.reduced_map[c] = 'cat'

    if FLAGS.salience:
        log('in gen salience!')
        root = DATA_FOLDER.resolve('ImageNet/output_tf')
        filenames = root.glob('train*').map(lambda x: x.abspath).tolist()  # validation
        import tensorflow as tf
        ds = tf.data.TFRecordDataset(filenames)
        #     for subroot in root:
        #         for imgfile in subroot:

        image_feature_description = {
            'image/height'      : tf.io.FixedLenFeature([], tf.int64),
            'image/width'       : tf.io.FixedLenFeature([], tf.int64),
            'image/colorspace'  : tf.io.FixedLenFeature([], tf.string),
            'image/channels'    : tf.io.FixedLenFeature([], tf.int64),
            'image/class/label' : tf.io.FixedLenFeature([], tf.int64),
            'image/class/synset': tf.io.FixedLenFeature([], tf.string),
            'image/class/text'  : tf.io.FixedLenFeature([], tf.string),
            # 'image/object/bbox/xmin' : tf.io.FixedLenFeature([], tf.float32),
            # 'image/object/bbox/xmax' : tf.io.FixedLenFeature([], tf.float32),
            # 'image/object/bbox/ymin' : tf.io.FixedLenFeature([], tf.float32),
            # 'image/object/bbox/ymax' : tf.io.FixedLenFeature([], tf.float32),
            # 'image/object/bbox/label': tf.io.FixedLenFeature([], tf.int64),
            'image/format'      : tf.io.FixedLenFeature([], tf.string),
            'image/filename'    : tf.io.FixedLenFeature([], tf.string),
            'image/encoded'     : tf.io.FixedLenFeature([], tf.string),
        }
        # imap = {}
        # current_i = -1
        # def input_gen():
        log('looping imagenet')

        _IMAGES_FOLDER[f'Training/{FLAGS.REGEN_NTRAIN}'].mkdirs()
        _IMAGES_FOLDER['Testing'].mkdirs()

        # classes = [
        #     'barn spider',
        #     'garden spider',
        #     'black widow',
        #     'wolf spider',
        #     'black and gold garden spider',
        #
        #     'emmet' ,#ant
        #     'grasshopper',
        #     'cricket',
        #     'stick insect',
        #     'cockroach'
        # ]



        class_count = {cn: 0 for cn in classes}

        for i, raw_record in enum(ds):
            example = tf.io.parse_single_example(raw_record, image_feature_description)
            # r[f'tf']['y_true'][i] = example['image/class/label'].numpy()
            # return tf.image.decode_jpeg(example['image/encoded'], channels=3).numpy()
            # if example['image/height'] != example['image/width']:
            #     continue

            if i % 100 == 0:
                log(f'on image {i}')
            classname = utf_decode(example['image/class/text'].numpy())
            for cn in classes:
                if (cn in classname) and (
                        class_count[cn] < (FLAGS.REGEN_NTRAIN if cn in not_trained else (FLAGS.REGEN_NTRAIN * 2))):
                    log(f'saving {cn} {class_count[cn] + 1}')
                    rrr = tf.image.decode_jpeg(example['image/encoded'], channels=3).numpy()
                    if class_count[cn] < FLAGS.REGEN_NTRAIN:
                        _IMAGES_FOLDER['Testing'][cn][f'{i}.png'].save(rrr)
                    else:
                        _IMAGES_FOLDER[f'Training/{FLAGS.REGEN_NTRAIN}']['dog' if cn in dogs else 'cat'][
                            f'{i}.png'].save(rrr)
                    class_count[cn] += 1
                    break
            break_all = True
            for cn, cc in listitems(class_count):
                if (cn in not_trained and cc != FLAGS.REGEN_NTRAIN) or (
                        cn not in not_trained and cc != (FLAGS.REGEN_NTRAIN * 2)):
                    break_all = False
            if break_all:
                break

            # current_i = current_i + 1
            # imap[i] = rrr
            # yield rrr
        # igen = input_gen()

        # def get_input(index):
        #     # log(f'trying to get index {index}')
        #     # log(f'current indices range from {safemin(list(imap.keys()))} to {safemax(list(imap.keys()))}')
        #     if index not in imap:
        #         # log('coud not get it')
        #         next(igen)
        #         return get_input(index)
        #     else:
        #         # log('got it!')
        #         rr = imap[index]
        #         for k in list(imap.keys()):
        #             if k < index:
        #                 del imap[k]
        #         return rr
        #     # for raw_record in ds:
        #     #     example = tf.io.parse_single_example(raw_record, image_feature_description)
        #     #     r[f'tf']['y_true'][index] = example['image/class/label'].numpy()
        #     #     return tf.image.decode_jpeg(example['image/encoded'], channels=3).numpy()
        #     # yield example
        # # y_true = []
        # # ifs_for_labels = input_files()
        # # for i in range(SANITY_SET.num):
        # #     y_true.append(next(ifs_for_labels)['image/class/label'].numpy())
        # # r[f'tf']['y_true'] = y_true
        # # def input_file_raws():
        # #     gen = input_files()
        # #     for example in gen:
        # #         yield tf.image.decode_jpeg(example['image/encoded'], channels=3).numpy()
        # # IN_files = input_file_raws()
        # IN_files = get_input


        # test_class_pairs = [
        #     pair for pair in chain(*[
        #         (
        #             SymAsymClassPair(n, False),
        #             SymAsymClassPair(n, True)
        #         ) for n in ints(np.linspace(0, 10, 6))
        #     ])
        # ]
        # class_pairs = [
        #     SymAsymClassPair(0, False),
        #     SymAsymClassPair(4, False)
        # ]
        # human_class_pairs = [
        #     SymAsymClassPair(0, False),
        #     SymAsymClassPair(2, False),
        #     SymAsymClassPair(4, False),
        #     SymAsymClassPair(6, False),
        #     SymAsymClassPair(8, False)
        # ]
        # gen_cfg = FLAGS.cfg_cfg['gen_cfg']
        # gen_images(
        #     folder=HUMAN_IMAGE_FOLDER['TimePilot'],
        #     class_pairs=human_class_pairs,
        #     ims_per_class=10
        # )

    else:
        test_class_pairs = [
            pair for pair in chain(*[
                (
                    SymAsymClassPair(n, False),
                    SymAsymClassPair(n, True)
                ) for n in ints(np.linspace(0, 10, 6))
            ])
        ]
        class_pairs = [
            SymAsymClassPair(0, False),
            SymAsymClassPair(4, False)
        ]
        human_class_pairs = [
            SymAsymClassPair(0, False),
            SymAsymClassPair(2, False),
            SymAsymClassPair(4, False),
            SymAsymClassPair(6, False),
            SymAsymClassPair(8, False)
        ]

        gen_images(
            folder=HUMAN_IMAGE_FOLDER['TimePilot'],
            class_pairs=human_class_pairs,
            ims_per_class=10
        )
        gen_images(
            folder=_IMAGES_FOLDER['RSA'],
            class_pairs=test_class_pairs,
            ims_per_class=10,
            # ims_per_class=1
        )
        gen_images(
            folder=_IMAGES_FOLDER['Testing'],
            class_pairs=test_class_pairs,
            ims_per_class=10,
            # ims_per_class=500,
            # ims_per_class=1
        )
        # for n in (25, 50, 100, 150, 200, 1000):
        for n in (10,):
            gen_images(
                folder=_IMAGES_FOLDER['Training'][n],
                class_pairs=class_pairs,
                ims_per_class=n
            )

    log('doing thing with _temp_ims')
    with mlib.file.TempFolder('_temp_ims') as temp:
        log('temp_ims_1')
        if temp.exists and temp.isdir:
            temp.clear()
        log('temp_ims_2')
        temp.mkdirs()
        log('temp_ims_3')
        [_IMAGES_FOLDER.copy_to(temp[f'gpu{i + 1}']) for i in range(gen_cfg['num_gpus'])]
        log('temp_ims_4')
        _IMAGES_FOLDER.clear()
        log('temp_ims_5')
        [temp[f'gpu{i + 1}'].moveinto(_IMAGES_FOLDER) for i in range(gen_cfg['num_gpus'])]
        log('temp_ims_6')
    log('finished thing with _temp_ims')
    nn_init_fun.NRC_IS_FINISHED()  # must be invoked this way since value of function changes
