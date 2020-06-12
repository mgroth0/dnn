# https://github.com/heuritech/convnets-keras

import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from imageio import imread
from PIL import Image

from lib.boot.loggy import log
import lib.nn.symnet as symnet
from lib.nn.arch.convnets.customlayers import splittensor, crosschannelnormalization

class AlexNet(symnet.SymNet):
    def ARCH_LABEL(cls): return "ALEX"
    @classmethod
    def HEIGHT_WIDTH(self):
        return 227
    def __init__(self, max_num_classes=2,batch_normalize=False, weights_path=symnet.data_folder.resolve("alexnet_weights_permute.h5").abspath):
        self.weights_path = weights_path
        super(AlexNet, self).__init__(max_num_classes,batch_normalize)


    def build_model(self):
        log('adding alexnet layers')
        inputs = Input(shape=(227, 227, 3))

        the_first_conv = self.Conv2DB(96, (11, 11), strides=4, activation='relu',name='conv_1')

        conv_1 = the_first_conv(inputs)

        conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
        conv_2 = crosschannelnormalization(name='convpool_1')(conv_2)
        conv_2 = ZeroPadding2D((2, 2))(conv_2)
        conv_2 = concatenate([
            self.Conv2DB(128, (5, 5), activation='relu', name='conv_2_' + str(i + 1))(
                splittensor(ratio_split=2, id_split=i)(conv_2)
            ) for i in range(2)], axis=3, name='conv_2')

        conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
        conv_3 = crosschannelnormalization()(conv_3)
        conv_3 = ZeroPadding2D((1, 1))(conv_3)
        conv_3 = self.Conv2DB(384, (3, 3), activation='relu', name='conv_3')(conv_3)

        conv_4 = ZeroPadding2D((1, 1))(conv_3)
        conv_4 = concatenate([
            self.Conv2DB(192, (3, 3), activation='relu', name='conv_4_' + str(i + 1))(
                splittensor(ratio_split=2, id_split=i)(conv_4)
            ) for i in range(2)], axis=3, name='conv_4')

        conv_5 = ZeroPadding2D((1, 1))(conv_4)
        conv_5 = concatenate([
            self.Conv2DB(128, (3, 3), activation='relu', name='conv_5_' + str(i + 1))(
                splittensor(ratio_split=2, id_split=i)(conv_5)
            ) for i in range(2)], axis=3, name='conv_5')

        dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)

        dense_1 = Flatten(name='flatten')(dense_1)
        dense_1 = self.DenseB(4096, activation='relu', name='dense_1')(dense_1)

        dense_2 = Dropout(0.5)(dense_1)
        dense_2 = self.DenseB(4096, activation='relu', name='dense_2')(dense_2)

        dense_3 = Dropout(0.5)(dense_2)
        dense_3 = self.DenseB(1000, name='dense_3')(dense_3)

        prediction = Activation('softmax', name='softmax')(dense_3)

        log('constructing alexnet model')
        model = Model(inputs, prediction)


        if self.weights_path:
            log('loading alexnet weights')
            model.load_weights(self.weights_path)
        log('returning alexnet model')
        return model

    def preprocess_image_batch(self, image_paths, img_size=None, crop_size=None, color_mode='rgb', out=None):
        """
        Consistent preprocessing of images batches

        :param image_paths: iterable: images to process
        :param crop_size: tuple: crop images if specified
        :param img_size: tuple: resize images if specified
        :param color_mode: Use rgb or change to bgr mode based on type of model you want to use
        :param out: append output to this iterable if specified
        """
        img_list = []

        for im_path in image_paths:
            img = imread(im_path, mode='RGB')
            if img_size:
                img = np.array(Image.fromarray(img).resize(img_size))

            img = img.astype('float32')
            # We normalize the colors (in RGB space) with the empirical means on the training set
            img[:, :, 0] -= 123.68
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 103.939
            # We permute the colors to get them in the BGR order
            if color_mode == 'bgr':
                img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
            img = img.transpose((2, 0, 1))

            if crop_size:
                img = img[:, (img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2
                , (img_size[1] - crop_size[1]) // 2:(img_size[1] + crop_size[1]) // 2]

            img_list.append(img)

        try:
            img_batch = np.stack(img_list, axis=0)
        except:
            raise ValueError('when img_size and crop_size are None, images'
                             ' in image_paths must have the same shapes.')

        if out is not None and hasattr(out, 'append'):
            out.append(img_batch)
        else:
            return img_batch
