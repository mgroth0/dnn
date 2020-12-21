from mlib.boot.mlog import err
print('nn_main.py: top')
from itertools import chain
import numpy as np
print('nn_main.py: about to do arch imports')
from arch import ALEX, GNET, INC, SCRATCH, AssembledModel
from arch.INC_ORIG import INC_ORIG
from arch.PROTO import PROTO
print('nn_main.py: finished arch imports')
from lib.nn.nnstate import reset_global_met_log
from lib.nn import nn_plotting, nnstate
from lib.nn.gen_preproc_ims import NN_Data_Dir, load_and_preprocess_ims, SymAsymClassPair, gen_images
from lib.dnn_data_saving import saveTestValResults, EXP_FOLDER
from lib.boot import nn_init_fun
from mlib.analyses import ANALYSES, AnalysisMode
print('nn_main.py: halfway through imports')
from mlib.boot import log
from mlib.boot.lang import listkeys, enum
from mlib.boot.stream import ints, listitems
from mlib.file import TempFolder, Folder
from mlib.proj.struct import pwdf
from mlib.str import utf_decode
from mlib.term import log_invokation
print('nn_main.py: done with imports')

ARCH_MAP = {
    'ALEX'    : ALEX,
    'GNET'    : GNET,
    'INC'     : INC,
    'INC_ORIG': INC_ORIG,
    'SCRATCH' : SCRATCH
}

# breakpoint()

import tensorflow as tf
@log_invokation()
def nnet_main(FLAGS):
    if FLAGS.salience:
        log('salience was here!')

    _IMAGES_FOLDER = pwdf()['_images'].mkdirs(mker=True)
    HUMAN_IMAGE_FOLDER = pwdf()['_images_human'].mkdirs(mker=True)

    cats = ['Egyptian cat',
            'Siamese cat',
            'Persian cat',
            'tiger cat',
            'tabby cat']
    dogs = [
        'Afghan hound',
        'basset hound',
        'beagle',
        'bloodhound',
        'bluetick'
    ]
    classes = cats + dogs
    not_trained = ['tabby cat', 'bluetick']
    for d in dogs:
        nnstate.reduced_map[d] = 'dog'
    for c in cats:
        nnstate.reduced_map[c] = 'cat'

    if FLAGS.gen:
        log('in gen!')
        _IMAGES_FOLDER.clearIfExists()
        HUMAN_IMAGE_FOLDER.clearIfExists()
        gen_cfg = FLAGS.cfg_cfg['gen_cfg']

        if FLAGS.salience:
            log('in gen salience!')
            root = Folder('/matt/data/ImageNet/output_tf')
            filenames = root.glob('train*').map(lambda x: x.abspath).tolist()  # validation
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

            _IMAGES_FOLDER[f'Training/{FLAGS.ntrain}'].mkdirs()
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

                if i % 100 == 0:
                    log(f'on image {i}')
                classname = utf_decode(example['image/class/text'].numpy())
                for cn in classes:
                    if (cn in classname) and (class_count[cn] < (FLAGS.ntrain if cn in not_trained else (FLAGS.ntrain*2))):
                        log(f'saving {cn} {class_count[cn] + 1}')
                        rrr = tf.image.decode_jpeg(example['image/encoded'], channels=3).numpy()
                        if class_count[cn] < FLAGS.ntrain:
                            _IMAGES_FOLDER['Testing'][cn][f'{i}.png'].save(rrr)
                        else:
                            _IMAGES_FOLDER[f'Training/{FLAGS.ntrain}']['dog' if cn in dogs else 'cat'][f'{i}.png'].save(rrr)
                        class_count[cn] += 1
                        break
                break_all = True
                for cn, cc in listitems(class_count):
                    if (cn in not_trained and cc != FLAGS.ntrain) or (cn not in not_trained and cc != (FLAGS.ntrain*2)):
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
        with TempFolder('_temp_ims') as temp:
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

    GPU_IMAGES_FOLDER = _IMAGES_FOLDER[f'gpu{FLAGS.mygpufordata}']

    GPU_TRAIN_FOLDER = NN_Data_Dir(GPU_IMAGES_FOLDER[f'Training/{FLAGS.ntrain}'])
    GPU_TEST_FOLDER = NN_Data_Dir(GPU_IMAGES_FOLDER[f'Testing'])
    GPU_RSA_FOLDER = NN_Data_Dir(GPU_IMAGES_FOLDER[f'RSA'])

    breakpoint()
    if FLAGS.deletenorms:
        GPU_TRAIN_FOLDER.delete_norm_dir()
        GPU_TEST_FOLDER.delete_norm_dir()
        GPU_RSA_FOLDER.delete_norm_dir()
        nn_init_fun.NRC_IS_FINISHED()  # must be invoked this way since value of function changes

    if FLAGS.normtrainims:
        err('im doing this?')

    nnstate.use_reduced_map = len(GPU_TRAIN_FOLDER.files) != len(GPU_TEST_FOLDER.files)
    datasetTrain, _ = load_and_preprocess_ims(
        TRAIN_TEST_SPLIT=1,
        data_dir=GPU_TRAIN_FOLDER,
        normalize_single_images=FLAGS.normtrainims)
    _, datasetVal = load_and_preprocess_ims(
        TRAIN_TEST_SPLIT=0,
        data_dir=GPU_TEST_FOLDER,
        normalize_single_images=FLAGS.normtrainims)
    _, datasetTest = load_and_preprocess_ims(
        TRAIN_TEST_SPLIT=0,
        data_dir=GPU_RSA_FOLDER,
        normalize_single_images=FLAGS.normtrainims
    )

    if FLAGS.proto_model:
        net = PROTO()
    else:
        net = ARCH_MAP[FLAGS.arch](
            max_num_classes=len(listkeys(datasetTest.class_label_map))
        )
    net.build(FLAGS)
    [a.after_build(FLAGS, net) for a in ANALYSES(mode=AnalysisMode.PIPELINE)]

    net.train_data = datasetTrain.prep(net.HEIGHT_WIDTH, net.PP)
    net.val_data = datasetVal.prep(net.HEIGHT_WIDTH, net.PP)
    net.test_data = datasetTest.prep(net.HEIGHT_WIDTH, net.PP)

    return trainTestRecord(net, '', FLAGS.epochs)


def trainTestRecord(net: AssembledModel, nam, nepochs):
    mets_for_each_epoch = []
    nnstate.step_counter = 1

    import lib.nn.net_mets as net_mets

    reset_global_met_log()
    for i in range(nepochs):
        nam = 'train'
        nnstate.MET_PHASE = 'epoch' + str(i + 1) + ':fit'
        if 'TRAIN' in nnstate.FLAGS.pipeline:
            net_mets.total_steps = net.train_data.num_steps  # len(net.train_data)
            # breakpoint()
            net_mets.batch_count = 0
            if isinstance(net, GNET):
                net_mets.batch_sub_count = 1
            else:
                net_mets.batch_sub_count = None

            net.train()
            log(f'finished another fit epoch!({i + 1}/{nepochs})')

            [a.after_fit(i, net, nam) for a in ANALYSES(mode=AnalysisMode.PIPELINE)]

        nnstate.MET_PHASE = 'epoch' + str(i + 1) + ':eval'
        if nnstate.EVAL_AND_REC_EVERY_EPOCH or i == nepochs - 1:
            if 'VAL' in nnstate.FLAGS.pipeline:
                net_mets.total_steps = net.val_data.num_steps  # len(net.val_data)
                net_mets.batch_count = 0
                if isinstance(net, GNET):
                    net_mets.batch_sub_count = 1
                else:
                    net_mets.batch_sub_count = None

                mets = net.val_eval()
                mets_for_each_epoch.append(mets)
                nam = 'val'
                saveTestValResults(net.ARCH_LABEL, nam, net.val_data, i)

                [a.after_val(i, net, nam) for a in ANALYSES(mode=AnalysisMode.PIPELINE)]

            nnstate.MET_PHASE = None
            nam = 'test'
            if 'REC' in nnstate.FLAGS.pipeline:
                net_mets.total_steps = net.test_data.num_steps  # len(net.test_data)
                net_mets.batch_count = 0
                if isinstance(net, GNET):
                    net_mets.batch_sub_count = 1
                else:
                    net_mets.batch_sub_count = None

                net.test_record(i)
                saveTestValResults(net.ARCH_LABEL, nam, net.test_data, i)

        log('Done with epoch $.', i + 1)
    old_name = nam
    from lib.nn import net_mets
    if 'TRAIN' in nnstate.FLAGS.pipeline and 'VAL' in nnstate.FLAGS.pipeline:
        for met in net_mets.METS_TO_USE():
            nam = met.__name__
            if met == net_mets.fill_cmat: continue
            nn_plotting.plot_metric(nam, nnstate.GLOBAL_MET_LOG[nam], old_name)

    return EXP_FOLDER()
