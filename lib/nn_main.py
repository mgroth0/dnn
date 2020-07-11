from itertools import chain
from keras_preprocessing.image import ImageDataGenerator
from numpy import ones
import tensorflow as tf
from tensorflow.keras.optimizers import SGD

from arch.GNET import PoolHelper, LRN
from lib.misc.imutil import resampleim
from mlib.boot.mutil import *
from lib.nn import nn_plotting, nnstate
from lib.nn.gen_preproc_ims import NN_Data_Dir, load_and_preprocess_ims, SymAsymClassPair, gen_images
from lib.data_saving import savePlotAndTableData, saveTestValResults, EXP_FOLDER
from lib.boot import nn_init_fun
from arch import *
from mlib.file import TempFolder, Folder, File
from mlib.proj.struct import pwdf
from mlib.term import log_invokation
@log_invokation()
def sym_net_main(FLAGS):
    _IMAGES_FOLDER = pwdf()['_images'].mkdirs(mker=True)
    HUMAN_IMAGE_FOLDER = pwdf()['_images_human'].mkdirs(mker=True)

    if FLAGS.gen:
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
        gen_cfg = FLAGS.cfg_cfg['gen_cfg']
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

        with TempFolder('_temp_ims') as temp:
            temp.mkdirs()
            [_IMAGES_FOLDER.copy_to(temp[f'gpu{i + 1}']) for i in range(gen_cfg['num_gpus'])]
            _IMAGES_FOLDER.clear()
            [temp[f'gpu{i + 1}'].moveinto(_IMAGES_FOLDER) for i in range(gen_cfg['num_gpus'])]
        nn_init_fun.NRC_IS_FINISHED()  # must be invoked this way since value of function changes

    GPU_IMAGES_FOLDER = _IMAGES_FOLDER[f'gpu{FLAGS.mygpufordata}']

    GPU_TRAIN_FOLDER = NN_Data_Dir(GPU_IMAGES_FOLDER[f'Training/{FLAGS.ntrain}'])
    GPU_TEST_FOLDER = NN_Data_Dir(GPU_IMAGES_FOLDER[f'Testing'])
    GPU_RSA_FOLDER = NN_Data_Dir(GPU_IMAGES_FOLDER[f'RSA'])

    if FLAGS.deletenorms:
        GPU_TRAIN_FOLDER.delete_norm_dir()
        GPU_TEST_FOLDER.delete_norm_dir()
        GPU_RSA_FOLDER.delete_norm_dir()
        nn_init_fun.NRC_IS_FINISHED()  # must be invoked this way since value of function changes

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

    net = {
        'ALEX'   : ALEX,
        'GNET'   : GNET,
        'INC'    : INC,
        'SCRATCH': SCRATCH
    }[FLAGS.arch](
        max_num_classes=len(listkeys(datasetTest.class_label_map)),
        proto=FLAGS.proto_model
    )
    net.build()
    net.train_data = datasetTrain.prep(net.META().HEIGHT_WIDTH)
    net.val_data = datasetVal.prep(net.META().HEIGHT_WIDTH)
    net.test_data = datasetTest.prep(net.META().HEIGHT_WIDTH)
    return trainTestRecord(net, '', FLAGS.epochs)


def trainTestRecord(net: SymNet, nam, nepochs):
    mets_for_each_epoch = []
    nnstate.step_counter = 1

    import lib.nn.net_mets as net_mets

    log('testing on 100 imagenet images')
    imagenet_test_images = Folder('_ImageNetTesting')
    log('here1')
    sanity_mat = None
    log('here2')
    sanity_key = []
    log('here3')
    INC_ORIG = False
    LOAD_DARIUS_MODEL = True
    if INC_ORIG:
        net.net = tf.keras.applications.InceptionResNetV2(
            include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
            pooling=None, classes=1000, classifier_activation='softmax'
        )
    elif LOAD_DARIUS_MODEL:
        if isinstance(net, ALEX):
            net.net = tf.keras.models.load_model('_data/darius_pretrained/alex_pretrained.h5')
        elif isinstance(net, GNET):
            net.net = tf.keras.models.load_model('_data/darius_pretrained/gnet_pretrained.h5',
                                                 custom_objects={'PoolHelper': PoolHelper, 'LRN': LRN})
        elif isinstance(net, INC):
            net.net = tf.keras.models.load_model('_data/darius_pretrained/inceptionresnetv2_pretrained.h5')

    # if isinstance(net, ALEX):
    #     sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #     net.net.compile(optimizer=sgd, loss='mse')
    # elif isinstance(net, GNET):
    #     sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #     net.net.compile(optimizer=sgd, loss='mse')
    with TempFolder('imagenet_test_images_temp') as f:
        log('here4')
        u = f['unknown'].mkdir()
        log('here5')
        for im in imagenet_test_images['unknown'].files:
            log('here6')
            # u.clear()
            # im.copy_into(u)
            # datagen = ImageDataGenerator(data_format=net.data_format()).flow_from_directory(
            #     f,
            #     target_size=(net.META().HEIGHT_WIDTH, net.META().HEIGHT_WIDTH),
            #     batch_size=1,
            #     shuffle=False
            # )


            img = im.load()
            if INC_ORIG:
                img = resampleim(img, 299, 299, nchan=3)
                img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)
            elif isinstance(net, ALEX):
                img = ALEX.preprocess_image(img, img_resize_height=256, img_resize_width=256, crop_size=(227, 227))
            elif isinstance(net, GNET):
                img = GNET.preprocess_image(img, img_resize_height=net.META().HEIGHT_WIDTH,
                                            img_resize_width=net.META().HEIGHT_WIDTH)
            else:
                img = resampleim(img, net.META().HEIGHT_WIDTH, net.META().HEIGHT_WIDTH, nchan=3)
                if isinstance(net, INC):
                    img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)

            if not INC_ORIG:
                data_format = net.data_format()
                if data_format == 'channels_first':
                    img = np.moveaxis(img, 2, 0)
            shape = [1] + list(img.shape)
            imdata = ones(tuple(shape))
            imdata[0, :, :, :] = img

            y_pred = net.net.predict(
                # datagen,
                imdata,
                # steps=steps,
                verbose=1,
                use_multiprocessing=True,
                # workers=16,
            )
            if isinstance(net, GNET):
                y_pred = y_pred[2]
            if sanity_mat is None:
                sanity_mat = y_pred
            else:
                sanity_mat = concat(sanity_mat, y_pred, axis=0)
            sanity_key += [im.name]
    if INC_ORIG:
        Folder('_data', mker=True)['sanity']['INC_ORIG']['ImageNetActivations.mat'].save({
            'activations': sanity_mat,
            'filenames'  : sanity_key
        })
    elif LOAD_DARIUS_MODEL:
        Folder('_data', mker=True)['sanity'][f'{net.META().ARCH_LABEL}_MAT_TO_PY']['ImageNetActivations.mat'].save({
            'activations': sanity_mat,
            'filenames'  : sanity_key
        })
    else:
        Folder('_data', mker=True)['sanity'][net.META().ARCH_LABEL]['ImageNetActivations.mat'].save({
            'activations': sanity_mat,
            'filenames'  : sanity_key
        })
    log('done with imagenet test')
    breakpoint()
    if INC_ORIG:
        err('no')

    for i in range(nepochs):
        nam = 'train'
        nnstate.MET_PHASE = 'epoch' + str(i + 1) + ':fit'
        if 'TRAIN' in nnstate.FLAGS.pipeline:
            net_mets.total_steps = len(net.train_data)
            net_mets.batch_count = 0
            if isinstance(net, GNET):
                net_mets.batch_sub_count = 1
            else:
                net_mets.batch_sub_count = None

            net.train()
            log(f'finished another fit epoch!({i + 1}/{nepochs})')
            if i == 0:
                log('saving examples')
                for ex in net.train_data.examples():
                    savePlotAndTableData(ex[1], nam, ex[0], isFigSet=False)
                log('finished saving examples')

        nnstate.MET_PHASE = 'epoch' + str(i + 1) + ':eval'
        if nnstate.EVAL_AND_REC_EVERY_EPOCH or i == nepochs - 1:
            if 'VAL' in nnstate.FLAGS.pipeline:
                net_mets.total_steps = len(net.val_data)
                net_mets.batch_count = 0
                if isinstance(net, GNET):
                    net_mets.batch_sub_count = 1
                else:
                    net_mets.batch_sub_count = None

                mets = net.val_eval()
                mets_for_each_epoch.append(mets)
                nam = 'val'
                saveTestValResults(net.META().ARCH_LABEL, nam, net.val_data, i)
                if i == 0:
                    log('saving examples')
                    exs = net.val_data.examples()
                    log(f'got {len(exs)} examples')
                    for idx, ex in enum(exs):
                        log(f'{idx=}')
                        savePlotAndTableData(ex[1], nam, ex[0], isFigSet=False)
                    log('finished saving examples')

            nnstate.MET_PHASE = None
            nam = 'test'
            if 'REC' in nnstate.FLAGS.pipeline:
                net_mets.total_steps = len(net.test_data)
                net_mets.batch_count = 0
                if isinstance(net, GNET):
                    net_mets.batch_sub_count = 1
                else:
                    net_mets.batch_sub_count = None

                net.test_record(i)
                saveTestValResults(net.META().ARCH_LABEL, nam, net.test_data, i)

        log('Done with epoch $.', i + 1)
    old_name = nam
    from lib.nn import net_mets
    if 'TRAIN' in nnstate.FLAGS.pipeline and 'VAL' in nnstate.FLAGS.pipeline:
        for met in net_mets.METS_TO_USE():
            nam = met.__name__
            if met == net_mets.fill_cmat: continue
            nn_plotting.plot_metric(nam, nnstate.GLOBAL_MET_LOG[nam], old_name)

    return EXP_FOLDER()
