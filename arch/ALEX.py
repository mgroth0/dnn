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

class ALEX(SymNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._next_dense_i = 1
    def META(self): return self.Meta(
        WEIGHTS='alexnet_weights_permute.h5',

        FULL_NAME='AlexNet',
        CREDITS='https://github.com/heuritech/convnets-keras',
        ARCH_LABEL='ALEX',
        HEIGHT_WIDTH=227

    )
    def assemble_layers(self):
        return Activation('softmax', name='softmax')(
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
                strides=2,
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
                        1
                    )(self.crosschannelnormalization(
                    )(self.max_pool(
                        strides=2
                    )(self._conv_group(
                        128,
                        5,
                        'conv_2',
                        zero_pad=2,
                        inputs=self.crosschannelnormalization(
                            name='convpool_1'
                        )(self.max_pool(
                            strides=3
                        )(self._conv(
                            96,
                            11,
                            strides=4,
                            name='conv_1'
                        )(self.inputs))))))))))))))))

    def max_pool(self, *args, **kwargs): return MaxPooling2D(3, *args, **kwargs)

    def _dense(self, *args, dropout=False, **kwargs):
        d = Dense(*args, name=f'dense_{self._next_dense_i}', **kwargs)
        if dropout:
            d = Dropout(0.5)(d)
        self._next_dense_i = self._next_dense_i + 1
        return d

    @staticmethod
    def crosschannelnormalization(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        # used in the original Alexnet
        def f(X):
            b, r, c, ch = X.shape
            half = n // 2
            square = K.square(X)
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
            scale = k
            for i in range(n):
                scale += alpha * extra_channels[:, :, :, i:i + ch]
            scale = scale**beta
            return X / scale

        return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)

    @staticmethod
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


    def _conv(self, n_filters, k_len, *args, **kwargs): return Conv2D(n_filters, (k_len, k_len), *args,
                                                                      activation="relu", **kwargs)
    def _cat(self, name): return Concatenate(axis=3, name=name)

    def _conv_group(self, n_filters, k_len, name, zero_pad, inputs):
        inputs = ZeroPadding2D(zero_pad)(inputs)
        return self._cat(
            name='name'
        )(
            [
                self._conv(
                    n_filters,
                    k_len,
                    name=f'{name}_{i + 1}'
                )(
                    self.splittensor(
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
