# -*- coding: utf-8 -*-
import pdb
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Concatenate

from lib.boot.loggy import log

def crosschannelnormalization(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet
    """

    def f(X):
        b, r, c, ch = X.shape
        half = n // 2
        square = K.square(X)





        # extra_channels = tf.keras.backend.spatial_2d_padding(
        ##     K.permute_dimensions(square, (0, 2, 3, 1))
            # square
            #                                   , ((0,half), (0,half)))

        extra_channels= tf.pad(
            square, paddings = tf.constant([[0, 0,], [0, 0],[0, 0,],[half, half,]]) )


        # extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))
        # scale = tf.cast(k, tf.float32)
        scale = k
        # orig_shape = extra_channels.shape
        for i in range(n):
            # log('i=$',i)
            # if i>0:
            #     newshape = extra_channels[:, :, :,i:i + ch].shape
            #     scale = tf.broadcast_to(scale,newshape)
            scale += alpha * extra_channels[:, :, :,i:i + ch]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)


def splittensor(axis=3, ratio_split=1, id_split=0, **kwargs):
    def f(X):
        div = X.shape[axis] // ratio_split

        if axis == 0:
            output = X[id_split * div:(id_split + 1) * div, :, :, :]
        elif axis == 1:
            output = X[:, id_split * div:(id_split + 1) * div, :, :]
        elif axis == 2:
            output = X[:, :, id_split * div:(id_split + 1) * div, :]
        elif axis == 3:
            output = X[:, :, :, id_split * div:(id_split + 1) * div]
        else:
            raise ValueError('This axis is not possible')

        return output

    def g(input_shape):
        output_shape = list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f, output_shape=lambda input_shape: g(input_shape), **kwargs)


def convolution2Dgroup(n_group, nb_filter, nb_row, nb_col, **kwargs):
    def f(input):
        return Concatenate(axis=1)([
                         Convolution2D(nb_filter // n_group, nb_row, nb_col)(
                             splittensor(axis=1,
                                         ratio_split=n_group,
                                         id_split=i)(input))
                         for i in range(n_group)
                         ])

    return f


class Softmax4D(Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def get_output_shape_for(self, input_shape):
        return input_shape
