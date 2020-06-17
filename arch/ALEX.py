from arch import *
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

class AlexNet(SymNet):
    def META(self): return self.Meta(
        WEIGHTS='alexnet_weights_permute.h5',

        FULL_NAME='AlexNet',
        CREDITS='https://github.com/heuritech/convnets-keras',
        ARCH_LABEL='ALEX',
        HEIGHT_WIDTH=227

    )
    def assemble_layers(self):
        conv_2 = ZeroPadding2D(
            (2, 2)
        )(self.crosschannelnormalization(
            name='convpool_1'
        )(MaxPooling2D(
            (3, 3),
            strides=(2, 2)
        )(Conv2D(
            96,
            (11, 11),
            strides=4,
            activation='relu',
            name='conv_1'
        )(self.inputs))))

        conv_4 = ZeroPadding2D(
            (1, 1)
        )(Conv2D(
            384,
            (3, 3),
            activation='relu',
            name='conv_3'
        )(ZeroPadding2D(
            (1, 1)
        )(self.crosschannelnormalization(

        )(MaxPooling2D(
            (3, 3),
            strides=(2, 2)
        )(Concatenate(
            axis=3,
            name='conv_2'
        )(
            [
                Conv2D(
                    128,
                    (5, 5),
                    activation='relu',
                    name=f'conv_2_{i + 1}'
                )(
                    self.splittensor(
                        ratio_split=2,
                        id_split=i
                    )(conv_2)
                ) for i in range(2)
            ]
        ))))))

        conv_5 = ZeroPadding2D(
            (1, 1)
        )(
            Concatenate(
                axis=3,
                name='conv_4'
            )(
                [
                    Conv2D(
                        192,
                        (3, 3),
                        activation='relu',
                        name=f'conv_4_{i + 1}'
                    )(
                        self.splittensor(
                            ratio_split=2,
                            id_split=i
                        )(conv_4)
                    ) for i in range(2)
                ]
            )
        )

        return Activation('softmax', name='softmax')(
            Dense(
                1000,
                name='dense_3'
            )(Dropout(
                0.5
            )(Dense(
                4096,
                activation='relu',
                name='dense_2'
            )(Dropout(
                0.5
            )(Dense(
                4096,
                activation='relu',
                name='dense_1'
            )(Flatten(
                name='flatten'
            )(MaxPooling2D(
                (3, 3),
                strides=(2, 2),
                name='convpool_5'
            )(Concatenate(
                axis=3,
                name='conv_5'
            )(
                [
                    Conv2D(
                        128,
                        (3, 3),
                        activation='relu',
                        name=f'conv_5_{i + 1}'
                    )(
                        self.splittensor(
                            ratio_split=2,
                            id_split=i
                        )(conv_5)
                    ) for i in range(2)
                ]
            )))))))))


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


    def convolution2Dgroup(self, n_group, nb_filter, nb_row, nb_col):
        def f(input_layer):
            return Concatenate(axis=1)([
                Conv2D(nb_filter // n_group, nb_row, nb_col)(
                    self.splittensor(axis=1,
                                     ratio_split=n_group,
                                     id_split=i)(input_layer))
                for i in range(n_group)
            ])

        return f


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
