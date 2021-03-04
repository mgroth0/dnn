import cv2
import matplotlib.image as mpimg
import os
import random

from lib.nn import nnstate
from lib.nn.tf_lib import Verbose
from mlib.boot import log
from mlib.boot.mlog import err
from mlib.term import log_invokation
SANITY_SWITCH = False
SANITY_MIX = True
import tensorflow as tf






@log_invokation(with_args=True)
def proko_train(
        model_class,
        epochs,
        HEIGHT_WIDTH,
        num_ims_per_class,
        include_top=True,  # THIS WAS THE BUG!!!! Probably used a different loss function while it was false
        weights=None,

        preprocess_class=None,
        loss='binary_crossentropy',
        classes=1,

):
    if num_ims_per_class < nnstate.FLAGS.batchsize:
        err('bad')
    print(f'starting script (num_ims_per_class={num_ims_per_class})')
    if classes == 1:
        net = model_class(
            include_top=include_top,
            weights=weights,  # 'imagenet' <- Proko used imagenet,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=classes,  # 1000,2
            classifier_activation='sigmoid',
            # layers=tf.keras.layers
        )
    else:
        net = model_class(
            include_top=include_top,
            weights=weights,  # 'imagenet' <- Proko used imagenet,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=classes,  # 1000,2
            # layers=tf.keras.layers
        )
    net.compile(
        optimizer='adam',
        loss=loss,
        metrics=['accuracy']
    )

    # HEIGHT_WIDTH = net.input.shape[0]


    # overfitting?
    # look at both accuracy and val accuracy!
    train_data, test_data = get_data(num_ims_per_class)

    print(f'starting training (num ims per class = {num_ims_per_class})')

    # if preprocess_class is None:
    #     preprocess_class = net
    #     print('getting preprocess_input from ' + str(model_class))
    #     preprocess_class = model_class

    ds = get_ds(train_data, HEIGHT_WIDTH, preprocess_class)
    test_ds = get_ds(test_data, HEIGHT_WIDTH, preprocess_class)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) == 0:
        err('no gpus')
    log('list of gpus:')
    for gpu in gpus:
        log(f'\tGPU:{gpu}')

    a_dict = {'history': None}
    def private_gpu_mem():
        print('starting private gpu mem')
        a_dict['history'] = net.fit(
            ds,
            epochs=epochs,
            verbose=Verbose.PROGRESS_BAR,
            use_multiprocessing=True,
            # use_multiprocessing=False,
            shuffle=False,
            validation_data=test_ds
        )
        print('starting testing')

        print(net.evaluate(
            ds,
            verbose=Verbose.PROGRESS_BAR,
            use_multiprocessing=True,
            # use_multiprocessing=False
        ))
        print('script complete')
        print('ending private gpu mem')

    # run_and_clear_gpu_mem_after(private_gpu_mem) #async memcpy from host to device: CUDA_ERROR_NOT_INITIALIZED: initialization error; GPU dst: 0xb041ef800; host src: 0x55b9b7f50dc0; size: 8=0x8

    private_gpu_mem()

    return a_dict['history']


