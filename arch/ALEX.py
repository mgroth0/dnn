from arch.shared import *
from tensorflow.keras.layers import (



    Concatenate,
    Dropout,
    Flatten,


    Lambda,
    Layer,
    ZeroPadding2D
)
from tensorflow import pad, constant
from tensorflow.keras import backend as K

from lib.misc.imutil import resampleim
from mlib.boot.mlog import err
_ALEX_CA = 3
class ALEX(SymNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._next_dense_i = 1

    def META(self): return self.Meta(
        WEIGHTS='alexnet_weights.h5' if _ALEX_CA == 1 else 'alexnet_weights_permute.h5',
        FULL_NAME='AlexNet',
        CREDITS='https://github.com/heuritech/convnets-keras',
        ARCH_LABEL='ALEX',
        HEIGHT_WIDTH=227,
        CHANNEL_AXIS=_ALEX_CA
    )
    def assemble_layers(self): return Activation('softmax', name='softmax')(
        self._dense(
            1000,
        )(self._dense(
            4096,
            activation='relu',
            dropout=True
        )(self._dense(
            4096,
            activation='relu',
            dropout=True
        )(Flatten(
            name='flatten'
        )(self.max_pool(
            name='convpool_5'
        )(self._conv_group(
            128,
            3,
            'conv_5',
            zero_pad=1,
            inputs=self._conv_group(
                192,
                3,
                'conv_4',
                zero_pad=1,
                inputs=self._conv(
                    384,
                    3,
                    name='conv_3'
                )(ZeroPadding2D(
                    1,
                    data_format=self.data_format()
                )(self.crosschannelnormalization(
                )(self.max_pool(
                )(self._conv_group(
                    128,
                    5,
                    'conv_2',
                    zero_pad=2,
                    inputs=self.crosschannelnormalization(
                        name='convpool_1'
                    )(self.max_pool(
                    )(self._conv(
                        96,
                        11,
                        strides=4,
                        name='conv_1'
                    )(self.inputs))))))))))))))))



    def max_pool(self, *args, **kwargs):
        return MaxPooling2D(3, *args, strides=2, data_format=self.data_format(), **kwargs)

    def _dense(self, *args, dropout=False, **kwargs):
        def f(inputs):
            d = Dense(*args, name=f'dense_{self._next_dense_i}', **kwargs)(inputs)
            print('_dense1')
            if dropout:
                try:
                    d = Dropout(0.5)(d)
                except:
                    print('except1')
                    breakpoint()
                    print('except2')
            self._next_dense_i = self._next_dense_i + 1
            return d
        return f

    # def _dense(self, *args, dropout=False, **kwargs):
    #     def f(inputs):
    #         d = Dense(*args, name=f'dense_{self._next_dense_i}', **kwargs)
    #     print('_dense1')
    #     if dropout:
    #         try:
    #             d = Dropout(0.5)(d)
    #         except:
    #             print('except1')
    #             print('except2')
    #     self._next_dense_i = self._next_dense_i + 1
    #     return d

    def crosschannelnormalization(self, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        # used in the original Alexnet
        def f(X):
            b = X.shape[0]
            r = X.shape[self.META().ROW_AXIS]
            c = X.shape[self.META().COL_AXIS]
            ch = X.shape[self.META().CHANNEL_AXIS]
            half = n // 2
            square = K.square(X)
            if self.CI == 0:
                extra_channels = pad(
                    square,
                    paddings=constant(
                        [
                            [0, 0],
                            [half, half],
                            [0, 0],
                            [0, 0]
                        ]
                    )
                )
            elif self.CI == 1:
                extra_channels = pad(
                    square,
                    paddings=constant(
                        [
                            [0, 0],
                            [0, 0],
                            [half, half],
                            [0, 0]
                        ]
                    )
                )
            elif self.CI == 2:
                extra_channels = pad(
                    square,
                    paddings=constant(
                        [
                            [0, 0],
                            [0, 0],
                            [0, 0],
                            [half, half]
                        ]
                    )
                )
            else:
                err(f'bad CI: {self.CI}')
            scale = k
            for i in range(n):
                if self.CI == 0:
                    scale += alpha * extra_channels[:, i:i + ch, :, :]
                elif self.CI == 1:
                    scale += alpha * extra_channels[:, :, i:i + ch, :]
                elif self.CI == 2:
                    scale += alpha * extra_channels[:, :, :, i:i + ch]
                else:
                    err(f'bad CI: {self.CI}')

            scale = scale**beta
            return X / scale

        return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)

    @staticmethod
    def splittensor(axis=1, ratio_split=1, id_split=0, **kwargs):
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


    def _conv(self, n_filters, k_len, *args, **kwargs):
        c = Conv2D(n_filters, (k_len, k_len), *args, activation="relu", data_format=self.data_format(), **kwargs)
        return c
    def _cat(self, name): return Concatenate(axis=self.CA, name=name)

    def _conv_group(self, n_filters, k_len, name, zero_pad, inputs):
        inputs = ZeroPadding2D(zero_pad, data_format=self.data_format())(inputs)
        return self._cat(
            name=name
        )(
            [
                self._conv(
                    n_filters,
                    k_len,
                    name=f'{name}_{i + 1}'
                )(
                    self.splittensor(
                        axis=self.CA,
                        ratio_split=2,
                        id_split=i
                    )(inputs)
                ) for i in range(2)
            ]
        )

    class Softmax4D(Layer):
        def __init__(self, axis=-1, **kwargs):
            self.axis = axis
            super().__init__(**kwargs)

        def build(self, input_shape):
            pass

        def call(self, x, mask=None):
            e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
            s = K.sum(e, axis=self.axis, keepdims=True)
            return e / s

        @staticmethod
        def get_output_shape_for(input_shape):
            return input_shape

    @staticmethod
    def preprocess_image(img, img_resize_height=None,img_resize_width=None, crop_size=None):
        # if img_size:
        img = resampleim(img, img_resize_height, img_resize_width, nchan=3)
        img = img.astype('float32')
        # We normalize the colors (in RGB space) with the empirical means on the training set
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        # We permute the colors to get them in the BGR order
        if crop_size:
            if _ALEX_CA == 1:
                img = img[:, (img_resize_height - crop_size[0]) // 2:(img_resize_height + crop_size[0]) // 2
                , (img_resize_width - crop_size[1]) // 2:(img_resize_width + crop_size[1]) // 2]
            else:
                img = img[(img_resize_height - crop_size[0]) // 2:(img_resize_height + crop_size[0]) // 2
                , (img_resize_width - crop_size[1]) // 2:(img_resize_width + crop_size[1]) // 2, :]
        return img
