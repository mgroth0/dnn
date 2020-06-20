from itertools import chain

from mlib.boot.mutil import *
from lib.nn import nn_plotting, nnstate
from lib.nn.gen_preproc_ims import NN_Data_Dir, load_and_preprocess_ims, SymAsymClassPair, gen_images
from lib.data_saving import savePlotAndTableData, saveTestValResults, EXP_FOLDER
from lib.boot import nn_init_fun
from arch import *
@log_invokation()
def sym_net_main(FLAGS):
    _IMAGES_FOLDER = pwdf()['_images']
    HUMAN_IMAGE_FOLDER = pwdf()['_images_human']

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
        human_class_pairs = [SymAsymClassPair(4, False)]
        gen_cfg = FLAGS.cfg_cfg['gen_cfg']
        _IMAGES_FOLDER = _IMAGES_FOLDER.deleteIfExists().mkdirs(mker=True)
        HUMAN_IMAGE_FOLDER = HUMAN_IMAGE_FOLDER.deleteIfExists().mkdirs(mker=True)
        gen_images(
            folder=HUMAN_IMAGE_FOLDER['TimePilot'],
            class_pairs=human_class_pairs,
            ims_per_class=50
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
            ims_per_class=500,
            # ims_per_class=1
        )
        for n in (25, 50, 100, 150, 200, 1000):
            # for n in (1,):
            gen_images(folder=_IMAGES_FOLDER['Training'][n], class_pairs=class_pairs, ims_per_class=n)

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
                for ex in net.train_data.examples():
                    savePlotAndTableData(ex[1], nam, ex[0], isFigSet=False)

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
                    for ex in net.val_data.examples():
                        savePlotAndTableData(ex[1], nam, ex[0], isFigSet=False)

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
