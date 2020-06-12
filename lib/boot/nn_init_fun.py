def setupTensorFlow(FLAGS):
    import os
    if FLAGS.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([c for c in FLAGS.gpus])
    import tensorflow as tf
    tf.random.set_seed(22)
    tf.compat.v1.enable_eager_execution()
    # by default all gpu mem is used. this option just makes it allocate less in the beginning and more as needed. Not sure why I would need this
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
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
    print('NRC IS FINISHED')
    import os
    os._exit(0)
