import logging
from types import ModuleType

from mlib.boot import log
from mlib.boot.mlog import err
from mlib.err import pub_print_warn

# MIN LOGS
# TF_LOG_DEVICE = False
# TF_LOG_CPP = '3'  # higher is fewer logs
# TF_LOG_DISABLE = True
# TF_LOG_LEVEL = logging.FATAL

TF_LOG_DEVICE = True
TF_LOG_CPP = '0'  # higher is fewer logs
TF_LOG_DISABLE = False
TF_LOG_LEVEL = logging.INFO

# MAX LOGS
# TF_LOG_DEVICE = True
# TF_LOG_CPP = '0'  # higher is fewer logs
# TF_LOG_DISABLE = False
# TF_LOG_LEVEL = logging.INFO

def setupTensorFlow(FLAGS=None) -> ModuleType:
    import os

    # This was central to how I managed to have multiple parallel jobs each with their own GPU. But I have to temporarily remove it to test if I can get gpus working in open mind at all
    # if FLAGS.gpus is not None:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([c for c in FLAGS.gpus])
    #

    # DEBUG
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_LOG_CPP  # BEFORE importing tf... now I think it worked!

    import tensorflow as tf
    tf.random.set_seed(22)
    tf.compat.v1.enable_eager_execution()
    # by default all gpu mem is used. this option just makes it allocate less in the beginning and more as needed. Not sure why I would need this
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) == 0:
        err('no gpus')
    log('list of gpus:')
    for gpu in gpus:
        log(f'\tGPU:{gpu}')
        # tf.config.esxperimental.set_memory_growth(gpu, True)

    tf.debugging.set_log_device_placement(TF_LOG_DEVICE)

    # trying to prevent endless allocation logging
    logger = tf.get_logger()
    logger.disabled = TF_LOG_DISABLE
    logger.setLevel(TF_LOG_LEVEL)

    # trying to prevent endless allocation logging
    # example: locator.cc:998] 1 Chunks of size 11645184 totalling 11.11MiB
    # 2021-02-27 17:51:27.248635: I tensorflow/core/common_runtime/bfc_allocator.cc:998] 1 Chunks of size 11645440 totalling 11.11MiB

    print(f'TF VERSION:{tf.__version__}')

    return tf

def runWithMultiProcess(main_nn_fun):
    pass
    # nn_main.MULTI = len(FLAGS.gpus) > 1
    # if nn_main.MULTI:
    #     strategy = tf.distribute.MirroredStrategy()
    #     # strategy = tf.distribute.experimental.CentralStorageStrategy()
    #     # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    #     with strategy.scope():
    #         if FLAGS.gen:
    #             nn_gen.nn_gen()
    #         else:
    #             sym_net_main()
    # else:


    # # BATCH_SIZE_PER_REPLICA = None  # to match darius
    #
    # BATCH_SIZE = None  # * len(
    # # BATCH_SIZE = BATCH_SIZE_PER_REPLICA  # * len(tf.config.list_physical_devices('GPU')) if MULTI_GPU else BATCH_SIZE_PER_REPLICA

def NRC_IS_FINISHED():
    # when imported, loggy upgrades this function to use log
    pub_print_warn()
    print('NRC IS FINISHED')

    # this used to be in loggy which is now in mlib
    from . import nn_init_fun
    def nrc_is_finished_upgrade(old):
        def f():
            log('NRC_IS_FINISHED')
            old()
        return f
    if nn_init_fun.NRC_IS_FINISHED.__name__ != 'f':
        nn_init_fun.NRC_IS_FINISHED = nrc_is_finished_upgrade(nn_init_fun.NRC_IS_FINISHED)

    import os
    os._exit(0)
