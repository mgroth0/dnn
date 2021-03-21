import random

from arch.proko_inc import NoBN_INC_PROKO
from mlib.boot.mlog import err
from mlib.boot.stream import listitems
from mlib.file import Folder
from rsa_for_darius import DATA_FOLDER
print('nn_main.py: top')
print('nn_main.py: about to do arch imports')
from arch import ALEX, GNET, INC, SCRATCH, AssembledModel
from arch.INC_ORIG import INC_ORIG
from arch.PROTO import PROTO
print('nn_main.py: finished arch imports')
from lib.nn.nnstate import reset_global_met_log
from lib.nn import nn_plotting, nnstate
from lib.nn.gen_preproc_ims import gen_main, NN_Data_Dir, load_and_preprocess_ims
from lib.dnn_data_saving import saveTestValResults, EXP_FOLDER
from lib.boot import nn_init_fun
from mlib.analyses import ANALYSES, AnalysisMode
print('nn_main.py: halfway through imports')
from mlib.boot import log
from mlib.boot.lang import listkeys
from mlib.proj.struct import pwdf
from mlib.term import log_invokation
print('nn_main.py: done with imports')

ARCH_MAP = {
    'ALEX'    : ALEX,
    'GNET'    : GNET,
    'INC'     : INC,
    'INC_ORIG': INC_ORIG,
    'SCRATCH' : SCRATCH,
    'CUSTOM'  : NoBN_INC_PROKO
}

# breakpoint()

@log_invokation()
def nnet_main(FLAGS):
    _IMAGES_FOLDER = pwdf()['_images'].mkdirs(mker=True)
    HUMAN_IMAGE_FOLDER = pwdf()['_images_human'].mkdirs(mker=True)

    if FLAGS.gen:
        gen_main(FLAGS, _IMAGES_FOLDER, HUMAN_IMAGE_FOLDER)
    if FLAGS.salience:
        class_map = {'dog': 0, 'cat': 1}

        # dogcatfolder = '/matt/data/tf_bug1/' #small set with hundreds I generated from imagenet
        dogcatfolder = DATA_FOLDER.resolve('tf_bug1/dogscats')  # thousands, downloaded from kaggle
        ntrain_folder = dogcatfolder['ntrain']
        dummy_folder = dogcatfolder['dummy'].mkdir()
        ntrain_folder.deleteIfExists().mkdir()
        for k, v in listitems(class_map):
            log('getting files')
            files = dogcatfolder['Training'][k].files.tolist()
            random.shuffle(files)
            log('looping files')
            for im in files[0:FLAGS.ntrain]:
                im.copyinto(ntrain_folder[k])

        GPU_TRAIN_FOLDER = NN_Data_Dir(ntrain_folder.abspath)
        GPU_TEST_FOLDER = NN_Data_Dir(dogcatfolder['Testing'].abspath)
        GPU_RSA_FOLDER = NN_Data_Dir(dummy_folder.abspath)
    else:
        GPU_IMAGES_FOLDER = _IMAGES_FOLDER[f'gpu{FLAGS.mygpufordata}']

        GPU_TRAIN_FOLDER = NN_Data_Dir(GPU_IMAGES_FOLDER[f'Training/{FLAGS.ntrain}'])
        GPU_TEST_FOLDER = NN_Data_Dir(GPU_IMAGES_FOLDER[f'Testing'])
        GPU_RSA_FOLDER = NN_Data_Dir(GPU_IMAGES_FOLDER[f'RSA'])

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
        # log(f'moving_mean(pre-train):{net.net.moving_mean}')
        # log(f'moving_var(pre-train):{net.net.moving_var}')
        if 'TRAIN' in nnstate.FLAGS.pipeline:
            nnstate.PIPELINE_PHASE = 'TRAIN'
            net_mets.total_steps = net.train_data.num_steps  # len(net.train_data)
            # breakpoint()
            net_mets.batch_count = 0
            if isinstance(net, GNET):
                net_mets.batch_sub_count = 1
            else:
                net_mets.batch_sub_count = None

            net.train()
            log(f'finished another fit epoch!({i + 1}/{nepochs})')

            # not sure why I didn't have this line in sym code any more
            saveTestValResults(net.ARCH_LABEL, nam, net.train_data, i)

            [a.after_fit(i, net, nam) for a in ANALYSES(mode=AnalysisMode.PIPELINE)]

        # log(f'moving_mean(post-train):{net.net.moving_mean}')
        # log(f'moving_var(post-train):{net.net.moving_var}')
        # breakpoint()
        nnstate.MET_PHASE = 'epoch' + str(i + 1) + ':eval'
        if nnstate.EVAL_AND_REC_EVERY_EPOCH or i == nepochs - 1:
            if 'VAL' in nnstate.FLAGS.pipeline:
                nnstate.PIPELINE_PHASE = 'VAL'
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
                nnstate.PIPELINE_PHASE = 'REC'
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

def run_and_clear_gpu_mem_after(lamb):
    # https://github.com/tensorflow/tensorflow/issues/36465
    import multiprocessing

    process_eval = multiprocessing.Process(target=lamb)
    process_eval.start()
    process_eval.join()
