from arch.assembled_model import AssembledModel
from mlib.boot.mlog import err

_ALEX_CA = 3
class ALEX(AssembledModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    # STATIC = AssembledModel.STATIC_ATTS(
    WEIGHTS = 'alexnet_weights.h5' if _ALEX_CA == 1 else 'alexnet_weights_permute.h5'
    FULL_NAME = 'AlexNet'
    CREDITS = 'https://github.com/heuritech/convnets-keras'
    ARCH_LABEL = 'ALEX'
    HEIGHT_WIDTH = 227
    CHANNEL_AXIS = _ALEX_CA
    # )
    def assemble_layers(self):
        self._next_dense_i = 1
        from tensorflow.keras.layers import (
            Dense,
            MaxPooling2D,
            Conv2D,
            Activation,
            Concatenate,
            Dropout,
            Flatten,
            Lambda,
            Layer,
            ZeroPadding2D
        )
        from tensorflow.keras import backend as K


        def max_pool(*args, **kwargs):
            return MaxPooling2D(3, *args, strides=2, data_format=self.data_format(), **kwargs)

        def _dense(*args, dropout=False, **kwargs):
            def f(inputs):
                d = Dense(*args, name=f'dense_{self._next_dense_i}', **kwargs)(inputs)
                print('_dense1')
                if dropout:
                    d = Dropout(0.5)(d)
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

        def crosschannelnormalization(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
            # from tensorflow import pad, constant
            # used in the original Alexnet
            def f(X):
                # err('')
                from tensorflow import constant


                # getting NameError: name 'pad' is not defined... what if I import here, does it help?
                from tensorflow import pad as padddddd

                b = X.shape[0]
                r = X.shape[self.ROW_AXIS]
                c = X.shape[self.COL_AXIS]
                ch = X.shape[self.CHANNEL_AXIS]
                half = n // 2
                square = K.square(X)
                if self.CI == 0:
                    extra_channels = padddddd(
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
                    extra_channels = padddddd(
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
                    extra_channels = padddddd(
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


            lamb = Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)

            return lamb

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


            lamb = Lambda(f, output_shape=lambda input_shape: g(input_shape), **kwargs)

            return lamb


        def _conv(n_filters, k_len, *args, **kwargs):
            c = Conv2D(n_filters, (k_len, k_len), *args, activation="relu", data_format=self.data_format(), **kwargs)
            return c
        def _cat(name): return Concatenate(axis=self.CA, name=name)

        def _conv_group(n_filters, k_len, name, zero_pad, inputs):
            inputs = ZeroPadding2D(zero_pad, data_format=self.data_format())(inputs)
            return _cat(
                name=name
            )(
                [
                    _conv(
                        n_filters,
                        k_len,
                        name=f'{name}_{i + 1}'
                    )(
                        splittensor(
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

        def get_output_shape_for(input_shape):
            return input_shape

        return Activation('softmax', name='softmax')(
            _dense(
                1000,
            )(_dense(
                4096,
                activation='relu',
                dropout=True
            )(_dense(
                4096,
                activation='relu',
                dropout=True
            )(Flatten(
                name='flatten'
            )(max_pool(
                name='convpool_5'
            )(_conv_group(
                128,
                3,
                'conv_5',
                zero_pad=1,
                inputs=_conv_group(
                    192,
                    3,
                    'conv_4',
                    zero_pad=1,
                    inputs=_conv(
                        384,
                        3,
                        name='conv_3'
                    )(ZeroPadding2D(
                        1,
                        data_format=self.data_format()
                    )(crosschannelnormalization(
                    )(max_pool(
                    )(_conv_group(
                        128,
                        5,
                        'conv_2',
                        zero_pad=2,
                        inputs=crosschannelnormalization(
                            name='convpool_1'
                        )(max_pool(
                        )(_conv(
                            96,
                            11,
                            strides=4,
                            name='conv_1'
                        )(self.inputs))))))))))))))))
