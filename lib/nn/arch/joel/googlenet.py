from __future__ import print_function

from tensorflow.keras.layers import Activation, AveragePooling2D, Concatenate, Dropout, Flatten, Input, \
    MaxPooling2D, ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils.conv_utils import convert_kernel

from lib.nn.arch.joel.lrn import LRN
from lib.nn.arch.joel.pool_helper import PoolHelper
from lib.nn.symnet import SymNet, data_folder

class GoogleNet(SymNet):
    def ARCH_LABEL(cls): return "GNET"
    @classmethod
    def HEIGHT_WIDTH(self):
        return 224
    INTER_LAY = -13
    def __init__(self, max_num_classes=2,weights_path=data_folder.resolve('googlenet_weights_permute.h5').abspath, batch_normalize=False):
        self.weights_path = weights_path
        super(GoogleNet, self).__init__(max_num_classes,batch_normalize)

    def build_model(self):
        # creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
        input = Input(shape=(224, 224, 3))

        input_pad = ZeroPadding2D(padding=(3, 3))(input)
        conv1_7x7_s2 = self.Conv2DB(64, (7, 7), strides=(2, 2), padding='valid', activation='relu', name='conv1/7x7_s2',
                                    kernel_regularizer=l2(0.0002))(input_pad)

        conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)

        pool1_helper = PoolHelper()(conv1_zero_pad)

        pool1_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool1/3x3_s2')(
            pool1_helper)

        pool1_norm1 = LRN(name='pool1/norm1')(pool1_3x3_s2)

        conv2_3x3_reduce = self.Conv2DB(64, (1, 1), padding='same', activation='relu', name='conv2/3x3_reduce',
                                        kernel_regularizer=l2(0.0002))(pool1_norm1)

        #########################
        #########################
        # cc = Conv2DB(64, (1,1), padding='same', activation='relu', name='conv2/3x3_reduce', kernel_regularizer=l2(0.0002))
        # cc(pool1_norm1)
        # cc.weights[0].shape
        #########################
        #########################
        conv2_3x3 = self.Conv2DB(192, (3, 3), padding='same', activation='relu', name='conv2/3x3',
                                 kernel_regularizer=l2(0.0002))(
            conv2_3x3_reduce)
        conv2_norm2 = LRN(name='conv2/norm2')(conv2_3x3)
        conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_norm2)
        pool2_helper = PoolHelper()(conv2_zero_pad)
        pool2_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2/3x3_s2')(
            pool2_helper)

        inception_3a_1x1 = self.Conv2DB(64, (1, 1), padding='same', activation='relu', name='inception_3a/1x1',
                                        kernel_regularizer=l2(0.0002))(pool2_3x3_s2)

        inception_3a_3x3_reduce = self.Conv2DB(96, (1, 1), padding='same', activation='relu',
                                               name='inception_3a/3x3_reduce',
                                               kernel_regularizer=l2(0.0002))(pool2_3x3_s2)

        inception_3a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3_reduce)
        inception_3a_3x3 = self.Conv2DB(128, (3, 3), padding='valid', activation='relu', name='inception_3a/3x3',
                                        kernel_regularizer=l2(0.0002))(inception_3a_3x3_pad)
        inception_3a_5x5_reduce = self.Conv2DB(16, (1, 1), padding='same', activation='relu',
                                               name='inception_3a/5x5_reduce',
                                               kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
        inception_3a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5_reduce)
        inception_3a_5x5 = self.Conv2DB(32, (5, 5), padding='valid', activation='relu', name='inception_3a/5x5',
                                        kernel_regularizer=l2(0.0002))(inception_3a_5x5_pad)
        inception_3a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_3a/pool')(
            pool2_3x3_s2)
        inception_3a_pool_proj = self.Conv2DB(32, (1, 1), padding='same', activation='relu',
                                              name='inception_3a/pool_proj',
                                              kernel_regularizer=l2(0.0002))(inception_3a_pool)
        inception_3a_output = Concatenate(axis=3, name='inception_3a/output')(
            [inception_3a_1x1, inception_3a_3x3, inception_3a_5x5, inception_3a_pool_proj])

        inception_3b_1x1 = self.Conv2DB(128, (1, 1), padding='same', activation='relu', name='inception_3b/1x1',
                                        kernel_regularizer=l2(0.0002))(inception_3a_output)
        inception_3b_3x3_reduce = self.Conv2DB(128, (1, 1), padding='same', activation='relu',
                                               name='inception_3b/3x3_reduce',
                                               kernel_regularizer=l2(0.0002))(inception_3a_output)
        inception_3b_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3_reduce)
        inception_3b_3x3 = self.Conv2DB(192, (3, 3), padding='valid', activation='relu', name='inception_3b/3x3',
                                        kernel_regularizer=l2(0.0002))(inception_3b_3x3_pad)
        inception_3b_5x5_reduce = self.Conv2DB(32, (1, 1), padding='same', activation='relu',
                                               name='inception_3b/5x5_reduce',
                                               kernel_regularizer=l2(0.0002))(inception_3a_output)
        inception_3b_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5_reduce)
        inception_3b_5x5 = self.Conv2DB(96, (5, 5), padding='valid', activation='relu', name='inception_3b/5x5',
                                        kernel_regularizer=l2(0.0002))(inception_3b_5x5_pad)
        inception_3b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_3b/pool')(
            inception_3a_output)
        inception_3b_pool_proj = self.Conv2DB(64, (1, 1), padding='same', activation='relu',
                                              name='inception_3b/pool_proj',
                                              kernel_regularizer=l2(0.0002))(inception_3b_pool)
        inception_3b_output = Concatenate(axis=3, name='inception_3b/output')(
            [inception_3b_1x1, inception_3b_3x3, inception_3b_5x5, inception_3b_pool_proj])

        inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_output)
        pool3_helper = PoolHelper()(inception_3b_output_zero_pad)
        pool3_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool3/3x3_s2')(
            pool3_helper)

        inception_4a_1x1 = self.Conv2DB(192, (1, 1), padding='same', activation='relu', name='inception_4a/1x1',
                                        kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
        inception_4a_3x3_reduce = self.Conv2DB(96, (1, 1), padding='same', activation='relu',
                                               name='inception_4a/3x3_reduce',
                                               kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
        inception_4a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4a_3x3_reduce)
        inception_4a_3x3 = self.Conv2DB(208, (3, 3), padding='valid', activation='relu', name='inception_4a/3x3',
                                        kernel_regularizer=l2(0.0002))(inception_4a_3x3_pad)
        inception_4a_5x5_reduce = self.Conv2DB(16, (1, 1), padding='same', activation='relu',
                                               name='inception_4a/5x5_reduce',
                                               kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
        inception_4a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4a_5x5_reduce)
        inception_4a_5x5 = self.Conv2DB(48, (5, 5), padding='valid', activation='relu', name='inception_4a/5x5',
                                        kernel_regularizer=l2(0.0002))(inception_4a_5x5_pad)
        inception_4a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4a/pool')(
            pool3_3x3_s2)
        inception_4a_pool_proj = self.Conv2DB(64, (1, 1), padding='same', activation='relu',
                                              name='inception_4a/pool_proj',
                                              kernel_regularizer=l2(0.0002))(inception_4a_pool)
        inception_4a_output = Concatenate(axis=3, name='inception_4a/output')(
            [inception_4a_1x1, inception_4a_3x3, inception_4a_5x5, inception_4a_pool_proj])

        loss1_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss1/ave_pool')(inception_4a_output)
        loss1_conv = self.Conv2DB(128, (1, 1), padding='same', activation='relu', name='loss1/conv',
                                  kernel_regularizer=l2(0.0002))(loss1_ave_pool)
        loss1_flat = Flatten()(loss1_conv)
        loss1_fc = self.DenseB(1024, activation='relu', name='loss1/fc', kernel_regularizer=l2(0.0002))(loss1_flat)
        loss1_drop_fc = Dropout(rate=0.7)(loss1_fc)
        loss1_classifier = self.DenseB(1000, name='loss1/classifier', kernel_regularizer=l2(0.0002))(loss1_drop_fc)
        loss1_classifier_act = Activation('softmax')(loss1_classifier)

        inception_4b_1x1 = self.Conv2DB(160, (1, 1), padding='same', activation='relu', name='inception_4b/1x1',
                                        kernel_regularizer=l2(0.0002))(inception_4a_output)
        inception_4b_3x3_reduce = self.Conv2DB(112, (1, 1), padding='same', activation='relu',
                                               name='inception_4b/3x3_reduce',
                                               kernel_regularizer=l2(0.0002))(inception_4a_output)
        inception_4b_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4b_3x3_reduce)
        inception_4b_3x3 = self.Conv2DB(224, (3, 3), padding='valid', activation='relu', name='inception_4b/3x3',
                                        kernel_regularizer=l2(0.0002))(inception_4b_3x3_pad)
        inception_4b_5x5_reduce = self.Conv2DB(24, (1, 1), padding='same', activation='relu',
                                               name='inception_4b/5x5_reduce',
                                               kernel_regularizer=l2(0.0002))(inception_4a_output)
        inception_4b_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4b_5x5_reduce)
        inception_4b_5x5 = self.Conv2DB(64, (5, 5), padding='valid', activation='relu', name='inception_4b/5x5',
                                        kernel_regularizer=l2(0.0002))(inception_4b_5x5_pad)
        inception_4b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4b/pool')(
            inception_4a_output)
        inception_4b_pool_proj = self.Conv2DB(64, (1, 1), padding='same', activation='relu',
                                              name='inception_4b/pool_proj',
                                              kernel_regularizer=l2(0.0002))(inception_4b_pool)
        inception_4b_output = Concatenate(axis=3, name='inception_4b/output')(
            [inception_4b_1x1, inception_4b_3x3, inception_4b_5x5, inception_4b_pool_proj])

        inception_4c_1x1 = self.Conv2DB(128, (1, 1), padding='same', activation='relu', name='inception_4c/1x1',
                                        kernel_regularizer=l2(0.0002))(inception_4b_output)
        inception_4c_3x3_reduce = self.Conv2DB(128, (1, 1), padding='same', activation='relu',
                                               name='inception_4c/3x3_reduce',
                                               kernel_regularizer=l2(0.0002))(inception_4b_output)
        inception_4c_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4c_3x3_reduce)
        inception_4c_3x3 = self.Conv2DB(256, (3, 3), padding='valid', activation='relu', name='inception_4c/3x3',
                                        kernel_regularizer=l2(0.0002))(inception_4c_3x3_pad)
        inception_4c_5x5_reduce = self.Conv2DB(24, (1, 1), padding='same', activation='relu',
                                               name='inception_4c/5x5_reduce',
                                               kernel_regularizer=l2(0.0002))(inception_4b_output)
        inception_4c_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4c_5x5_reduce)
        inception_4c_5x5 = self.Conv2DB(64, (5, 5), padding='valid', activation='relu', name='inception_4c/5x5',
                                        kernel_regularizer=l2(0.0002))(inception_4c_5x5_pad)
        inception_4c_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4c/pool')(
            inception_4b_output)
        inception_4c_pool_proj = self.Conv2DB(64, (1, 1), padding='same', activation='relu',
                                              name='inception_4c/pool_proj',
                                              kernel_regularizer=l2(0.0002))(inception_4c_pool)
        inception_4c_output = Concatenate(axis=3, name='inception_4c/output')(
            [inception_4c_1x1, inception_4c_3x3, inception_4c_5x5, inception_4c_pool_proj])

        inception_4d_1x1 = self.Conv2DB(112, (1, 1), padding='same', activation='relu', name='inception_4d/1x1',
                                        kernel_regularizer=l2(0.0002))(inception_4c_output)
        inception_4d_3x3_reduce = self.Conv2DB(144, (1, 1), padding='same', activation='relu',
                                               name='inception_4d/3x3_reduce',
                                               kernel_regularizer=l2(0.0002))(inception_4c_output)
        inception_4d_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4d_3x3_reduce)
        inception_4d_3x3 = self.Conv2DB(288, (3, 3), padding='valid', activation='relu', name='inception_4d/3x3',
                                        kernel_regularizer=l2(0.0002))(inception_4d_3x3_pad)
        inception_4d_5x5_reduce = self.Conv2DB(32, (1, 1), padding='same', activation='relu',
                                               name='inception_4d/5x5_reduce',
                                               kernel_regularizer=l2(0.0002))(inception_4c_output)
        inception_4d_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4d_5x5_reduce)
        inception_4d_5x5 = self.Conv2DB(64, (5, 5), padding='valid', activation='relu', name='inception_4d/5x5',
                                        kernel_regularizer=l2(0.0002))(inception_4d_5x5_pad)
        inception_4d_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4d/pool')(
            inception_4c_output)
        inception_4d_pool_proj = self.Conv2DB(64, (1, 1), padding='same', activation='relu',
                                              name='inception_4d/pool_proj',
                                              kernel_regularizer=l2(0.0002))(inception_4d_pool)
        inception_4d_output = Concatenate(axis=3, name='inception_4d/output')(
            [inception_4d_1x1, inception_4d_3x3, inception_4d_5x5, inception_4d_pool_proj])

        loss2_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss2/ave_pool')(inception_4d_output)
        loss2_conv = self.Conv2DB(128, (1, 1), padding='same', activation='relu', name='loss2/conv',
                                  kernel_regularizer=l2(0.0002))(loss2_ave_pool)
        loss2_flat = Flatten()(loss2_conv)
        loss2_fc = self.DenseB(1024, activation='relu', name='loss2/fc', kernel_regularizer=l2(0.0002))(loss2_flat)
        loss2_drop_fc = Dropout(rate=0.7)(loss2_fc)
        loss2_classifier = self.DenseB(1000, name='loss2/classifier', kernel_regularizer=l2(0.0002))(loss2_drop_fc)
        loss2_classifier_act = Activation('softmax')(loss2_classifier)

        inception_4e_1x1 = self.Conv2DB(256, (1, 1), padding='same', activation='relu', name='inception_4e/1x1',
                                        kernel_regularizer=l2(0.0002))(inception_4d_output)
        inception_4e_3x3_reduce = self.Conv2DB(160, (1, 1), padding='same', activation='relu',
                                               name='inception_4e/3x3_reduce',
                                               kernel_regularizer=l2(0.0002))(inception_4d_output)
        inception_4e_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_3x3_reduce)
        inception_4e_3x3 = self.Conv2DB(320, (3, 3), padding='valid', activation='relu', name='inception_4e/3x3',
                                        kernel_regularizer=l2(0.0002))(inception_4e_3x3_pad)
        inception_4e_5x5_reduce = self.Conv2DB(32, (1, 1), padding='same', activation='relu',
                                               name='inception_4e/5x5_reduce',
                                               kernel_regularizer=l2(0.0002))(inception_4d_output)
        inception_4e_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4e_5x5_reduce)
        inception_4e_5x5 = self.Conv2DB(128, (5, 5), padding='valid', activation='relu', name='inception_4e/5x5',
                                        kernel_regularizer=l2(0.0002))(inception_4e_5x5_pad)
        inception_4e_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4e/pool')(
            inception_4d_output)
        inception_4e_pool_proj = self.Conv2DB(128, (1, 1), padding='same', activation='relu',
                                              name='inception_4e/pool_proj',
                                              kernel_regularizer=l2(0.0002))(inception_4e_pool)
        inception_4e_output = Concatenate(axis=3, name='inception_4e/output')(
            [inception_4e_1x1, inception_4e_3x3, inception_4e_5x5, inception_4e_pool_proj])

        inception_4e_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_output)
        pool4_helper = PoolHelper()(inception_4e_output_zero_pad)
        pool4_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool4/3x3_s2')(
            pool4_helper)

        inception_5a_1x1 = self.Conv2DB(256, (1, 1), padding='same', activation='relu', name='inception_5a/1x1',
                                        kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
        inception_5a_3x3_reduce = self.Conv2DB(160, (1, 1), padding='same', activation='relu',
                                               name='inception_5a/3x3_reduce',
                                               kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
        inception_5a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_5a_3x3_reduce)
        inception_5a_3x3 = self.Conv2DB(320, (3, 3), padding='valid', activation='relu', name='inception_5a/3x3',
                                        kernel_regularizer=l2(0.0002))(inception_5a_3x3_pad)
        inception_5a_5x5_reduce = self.Conv2DB(32, (1, 1), padding='same', activation='relu',
                                               name='inception_5a/5x5_reduce',
                                               kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
        inception_5a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_5a_5x5_reduce)
        inception_5a_5x5 = self.Conv2DB(128, (5, 5), padding='valid', activation='relu', name='inception_5a/5x5',
                                        kernel_regularizer=l2(0.0002))(inception_5a_5x5_pad)
        inception_5a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_5a/pool')(
            pool4_3x3_s2)
        inception_5a_pool_proj = self.Conv2DB(128, (1, 1), padding='same', activation='relu',
                                              name='inception_5a/pool_proj',
                                              kernel_regularizer=l2(0.0002))(inception_5a_pool)
        inception_5a_output = Concatenate(axis=3, name='inception_5a/output')(
            [inception_5a_1x1, inception_5a_3x3, inception_5a_5x5, inception_5a_pool_proj])

        inception_5b_1x1 = self.Conv2DB(384, (1, 1), padding='same', activation='relu', name='inception_5b/1x1',
                                        kernel_regularizer=l2(0.0002))(inception_5a_output)
        inception_5b_3x3_reduce = self.Conv2DB(192, (1, 1), padding='same', activation='relu',
                                               name='inception_5b/3x3_reduce',
                                               kernel_regularizer=l2(0.0002))(inception_5a_output)
        inception_5b_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_5b_3x3_reduce)
        inception_5b_3x3 = self.Conv2DB(384, (3, 3), padding='valid', activation='relu', name='inception_5b/3x3',
                                        kernel_regularizer=l2(0.0002))(inception_5b_3x3_pad)
        inception_5b_5x5_reduce = self.Conv2DB(48, (1, 1), padding='same', activation='relu',
                                               name='inception_5b/5x5_reduce',
                                               kernel_regularizer=l2(0.0002))(inception_5a_output)
        inception_5b_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_5b_5x5_reduce)
        inception_5b_5x5 = self.Conv2DB(128, (5, 5), padding='valid', activation='relu', name='inception_5b/5x5',
                                        kernel_regularizer=l2(0.0002))(inception_5b_5x5_pad)
        inception_5b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_5b/pool')(
            inception_5a_output)
        inception_5b_pool_proj = self.Conv2DB(128, (1, 1), padding='same', activation='relu',
                                              name='inception_5b/pool_proj',
                                              kernel_regularizer=l2(0.0002))(inception_5b_pool)
        inception_5b_output = Concatenate(axis=3, name='inception_5b/output')(
            [inception_5b_1x1, inception_5b_3x3, inception_5b_5x5, inception_5b_pool_proj])

        pool5_7x7_s1 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), name='pool5/7x7_s2')(inception_5b_output)
        loss3_flat = Flatten()(pool5_7x7_s1)
        pool5_drop_7x7_s1 = Dropout(rate=0.4)(loss3_flat)
        loss3_classifier = self.DenseB(1000, name='loss3/classifier', kernel_regularizer=l2(0.0002))(pool5_drop_7x7_s1)
        loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)

        googlenet = Model(inputs=input, outputs=[loss1_classifier_act, loss2_classifier_act, loss3_classifier_act])

        if self.weights_path:
            googlenet.load_weights(self.weights_path)

        # convert the convolutional kernels for tensorflow
        for layer in googlenet.layers:
            if layer.__class__.__name__ == 'Conv2D':
                original_w = K.eval(layer.kernel)
                converted_w = convert_kernel(original_w)
                layer.kernel.assign(converted_w)

        return googlenet
