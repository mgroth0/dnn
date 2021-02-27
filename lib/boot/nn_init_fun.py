from types import ModuleType

from mlib.err import pub_print_warn
def setupTensorFlow(FLAGS=None) -> ModuleType:
    import os

    # This was central to how I managed to have multiple parallel jobs each with their own GPU. But I have to temporarily remove it to test if I can get gpus working in open mind at all
    # if FLAGS.gpus is not None:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([c for c in FLAGS.gpus])
    #
    import tensorflow as tf
    tf.random.set_seed(22)
    tf.compat.v1.enable_eager_execution()
    # by default all gpu mem is used. this option just makes it allocate less in the beginning and more as needed. Not sure why I would need this
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    tf.debugging.set_log_device_placement(True)
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

