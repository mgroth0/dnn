import tensorflow as tf

from lib import data_saving
from lib.gpu import mygpus
from arch.ALEX import AlexNet
from arch.INC import IRNV2
from arch.GNET import GoogleNet
from arch.SCRATCH import ScratchNet
from lib.nn import nn_plotting
from lib.nn.gen_preproc_ims import *
from lib.defaults import *
from lib.nn.nnstate import reset_global_met_log
from lib.data_saving import savePlotAndTableData, saveTestValResults
from arch.symnet import SymNet
from lib.boot import nn_init_fun
def sym_net_main(FLAGS):
    log('running sym_net_main')
    import lib.nn.nnstate as nnstate
    nnstate.FLAGS = FLAGS
    reset_global_met_log()
    normalize_single_images = nnstate.FLAGS.normtrainims

    BASE_IMAGE_FOLDER = File(os.getcwd()).resolve('_images')
    mygpufordata = mygpus()[0]+1 if not isempty(mygpus()) else 1
    MY_DATA_FOLDER = File(BASE_IMAGE_FOLDER.abspath + '/gpu' + str(mygpufordata))
    MY_TRAIN_FOLDER = NN_Data_Dir(f'{MY_DATA_FOLDER.abspath}/Training/{FLAGS.ntrain}')
    MY_TEST_FOLDER = NN_Data_Dir(f'{MY_DATA_FOLDER.abspath}/Testing')
    MY_RSA_FOLDER = NN_Data_Dir(f'{MY_DATA_FOLDER.abspath}/ForMatt')

    cfg_cfg = json.loads(FLAGS.cfg)
    root = cfg_cfg['root']
    data_saving.root = root

    if FLAGS.gen:
        from lib.nn import nn_gen
        gen_cfg = cfg_cfg['gen_cfg']
        nn_gen.nn_gen(FLAGS, BASE_IMAGE_FOLDER,
                      num_gpus=gen_cfg['num_gpus'],
                      TRAINING_SET_SIZES=gen_cfg['TRAINING_SET_SIZES'],
                      EVAL_SIZE=gen_cfg['EVAL_SIZE'],
                      RSA_SIZE_PER_CLASS=gen_cfg['RSA_SIZE_PER_CLASS']
                      )
        nn_init_fun.NRC_IS_FINISHED()  # must be invoked this way since value of function changes
    elif FLAGS.deletenorms:
        log('just deleting norm dirs...')
        MY_TRAIN_FOLDER.delete_norm_dir()
        MY_TEST_FOLDER.delete_norm_dir()
        MY_RSA_FOLDER.delete_norm_dir()
        log('deleted norm dirs!')
        nn_init_fun.NRC_IS_FINISHED()  # must be invoked this way since value of function changes
    datasetTrain, _ = load_and_preprocess_ims(1,
                                              data_dir=MY_TRAIN_FOLDER,
                                              normalize_single_images=normalize_single_images)
    _, datasetVal = load_and_preprocess_ims(0,
                                            data_dir=MY_TEST_FOLDER,
                                            normalize_single_images=normalize_single_images)
    _, datasetTest = load_and_preprocess_ims(0, data_dir=MY_RSA_FOLDER,
                                             normalize_single_images=normalize_single_images)

    def get_available_gpus():
        local_device_protos = tf.python.client.device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']


    net = {
        'ALEX'   : AlexNet,
        'GNET'   : GoogleNet,
        'INC'    : IRNV2,
        'SCRATCH': ScratchNet
    }[FLAGS.arch](
        # num_classes=len(band_groups[i]) * 2
        # ,
        batch_normalize={
            'ALEX'   : False,
            'GNET'   : False,
            'INC'    : True,
            'SCRATCH': True
        }[FLAGS.arch],
        max_num_classes=len(listkeys(datasetTest.class_label_map)),
        proto=len(get_available_gpus()) == 0
    )
    datasetTrain.prep(net.HEIGHT_WIDTH())
    datasetVal.prep(net.HEIGHT_WIDTH())
    datasetTest.prep(net.HEIGHT_WIDTH())
    net.train_data = datasetTrain
    net.val_data = datasetVal
    net.test_data = datasetTest
    trainTestRecord(net, '', FLAGS.epochs)
    loggy.log('finished sym_net_main')
    return data_saving.EXP_FOLDER(root)


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
            if isinstance(net, GoogleNet):
                net_mets.batch_sub_count = 1
            else:
                net_mets.batch_sub_count = None

            net.train()
            log(f'finished another fit epoch!({i + 1}/{nepochs})')
            if i == 0:
                for ex in net.train_data.examples(net.HEIGHT_WIDTH()):
                    savePlotAndTableData(ex[1], nam, ex[0], isFigSet=False)

        nnstate.MET_PHASE = 'epoch' + str(i + 1) + ':eval'
        if nnstate.EVAL_AND_REC_EVERY_EPOCH or i == nepochs - 1:
            if 'VAL' in nnstate.FLAGS.pipeline:
                net_mets.total_steps = len(net.val_data)
                net_mets.batch_count = 0
                if isinstance(net, GoogleNet):
                    net_mets.batch_sub_count = 1
                else:
                    net_mets.batch_sub_count = None

                mets = net.val_eval()
                mets_for_each_epoch.append(mets)
                nam = 'val'
                saveTestValResults(net.ARCH_LABEL(), nam, net.val_data, i)
                if i == 0:
                    for ex in net.val_data.examples(net.HEIGHT_WIDTH()):
                        savePlotAndTableData(ex[1], nam, ex[0], isFigSet=False)

            nnstate.MET_PHASE = None
            nam = 'test'
            if 'REC' in nnstate.FLAGS.pipeline:
                net_mets.total_steps = len(net.test_data)
                net_mets.batch_count = 0
                if isinstance(net, GoogleNet):
                    net_mets.batch_sub_count = 1
                else:
                    net_mets.batch_sub_count = None

                net.test_record(i)
                saveTestValResults(net.ARCH_LABEL(), nam, net.test_data, i)

        log('Done with epoch $.', i + 1)
    old_name = nam
    from lib.nn import net_mets
    if 'TRAIN' in nnstate.FLAGS.pipeline and 'VAL' in nnstate.FLAGS.pipeline:
        for met in net_mets.METS_TO_USE():
            nam = met.__name__
            if met == net_mets.fill_cmat: continue
            nn_plotting.plot_metric(nam, nnstate.GLOBAL_MET_LOG[nam], old_name)
