from arch.assembled_model import AssembledModel
from lib.misc.imutil import resampleim
from mlib.boot.mlog import err

class GNET(AssembledModel):
    # STATIC = AssembledModel.STATIC_ATTS(
    WEIGHTS = 'googlenet_weights_permute.h5'
    FLIPPED_CONV_WEIGHTS = True
    FULL_NAME = 'GoogleNet'
    CREDITS = 'joel,GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)'
    ARCH_LABEL = 'GNET'
    HEIGHT_WIDTH = 224
    INTER_LAY = -13
    OUTPUT_IDX = 2
    # )
    def assemble_layers(self):
        from tensorflow.keras.layers import (
            Dense,
            MaxPooling2D,
            Conv2D,
            Activation,
            AveragePooling2D,
            Concatenate,
            Dropout,
            Flatten,
            ZeroPadding2D
        )
        from tensorflow.keras.regularizers import l2
        LRN, PoolHelper = gnet_layer_classes()
        LRN: type
        PoolHelper: type

        def _dense(*args, **kwargs):
            return Dense(*args, **kwargs, kernel_regularizer=l2(0.0002))

        def _dense_c(n, *args, **kwargs):
            return _dense(1000, *args, name=f'loss{n}/classifier', **kwargs)



        def _conv2d(*args, **kwargs):
            # layers with .__class__.__name__ Conv2D have weights flipped. So don't change class name
            return Conv2D(*args, **kwargs, activation='relu', kernel_regularizer=l2(0.0002))

        def _act(**kwargs):
            return Activation('softmax', **kwargs)
        def preprocess_image(img, img_resize_height=None, img_resize_width=None, crop_size=None):
            err('use preprocessor')
            # if img_size:
            img = resampleim(img, img_resize_height, img_resize_width, nchan=3)
            img = img.astype('float32')
            # We normalize the colors (in RGB space) with the empirical means on the training set
            img[:, :, 0] -= 123.68
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 103.939
            # We permute the colors to get them in the BGR order
            # if crop_size:
            #     # if _ALEX_CA == 1:
            #     #     img = img[:, (img_resize_height - crop_size[0]) // 2:(img_resize_height + crop_size[0]) // 2
            #     #     , (img_resize_width - crop_size[1]) // 2:(img_resize_width + crop_size[1]) // 2]
            #     # else:
            #     img = img[(img_resize_height - crop_size[0]) // 2:(img_resize_height + crop_size[0]) // 2
            #     , (img_resize_width - crop_size[1]) // 2:(img_resize_width + crop_size[1]) // 2, :]
            return img



        pool2_3x3_s2 = MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid',
            name='pool2/3x3_s2'
        )(PoolHelper()(ZeroPadding2D(padding=(1, 1))(LRN(name='conv2/norm2')(_conv2d(
            192,
            (3, 3),
            padding='same',
            name='conv2/3x3'
        )(_conv2d(64, (1, 1), padding='same', name='conv2/3x3_reduce')(LRN(name='pool1/norm1')(MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid',
            name='pool1/3x3_s2'
        )(PoolHelper()(ZeroPadding2D(padding=(1, 1))(_conv2d(
            64,
            (7, 7),
            strides=(2, 2),
            padding='valid',
            name='conv1/7x7_s2'
        )(ZeroPadding2D(
            padding=(3, 3)
        )(self.inputs))))))))))))

        inception_3a_output = Concatenate(
            axis=3,
            name='inception_3a/output'
        )(
            [
                _conv2d(64, (1, 1), padding='same', name='inception_3a/1x1')(pool2_3x3_s2),
                _conv2d(128, (3, 3), padding='valid', name='inception_3a/3x3')(ZeroPadding2D(
                    padding=(1, 1)
                )(_conv2d(
                    96,
                    (1, 1),
                    padding='same',
                    name='inception_3a/3x3_reduce'
                )(pool2_3x3_s2))),
                _conv2d(32, (5, 5), padding='valid', name='inception_3a/5x5')(
                    ZeroPadding2D(padding=(2, 2))(_conv2d(
                        16,
                        (1, 1),
                        padding='same',
                        name='inception_3a/5x5_reduce'
                    )(pool2_3x3_s2))),
                _conv2d(
                    32,
                    (1, 1),
                    padding='same',
                    name='inception_3a/pool_proj'
                )(MaxPooling2D(
                    pool_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    name='inception_3a/pool'
                )(pool2_3x3_s2))
            ]
        )

        pool3_3x3_s2 = MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid',
            name='pool3/3x3_s2'
        )(PoolHelper()(ZeroPadding2D(padding=(1, 1))(Concatenate(
            axis=3,
            name='inception_3b/output'
        )([
            _conv2d(
                128,
                (1, 1),
                padding='same',
                name='inception_3b/1x1'
            )(inception_3a_output),
            _conv2d(
                192,
                (3, 3),
                padding='valid',
                name='inception_3b/3x3'
            )(ZeroPadding2D(
                padding=(1, 1)
            )(_conv2d(
                128,
                (1, 1),
                padding='same',
                name='inception_3b/3x3_reduce'
            )(inception_3a_output))),
            _conv2d(
                96,
                (5, 5),
                padding='valid',
                name='inception_3b/5x5'
            )(ZeroPadding2D(
                padding=(2, 2)
            )(_conv2d(
                32,
                (1, 1),
                padding='same',
                name='inception_3b/5x5_reduce'
            )(inception_3a_output))),
            _conv2d(
                64,
                (1, 1),
                padding='same',
                name='inception_3b/pool_proj'
            )(MaxPooling2D(
                pool_size=(3, 3),
                strides=(1, 1),
                padding='same',
                name='inception_3b/pool'
            )(inception_3a_output))
        ]))))

        inception_4a_output = Concatenate(axis=3, name='inception_4a/output')(
            [
                _conv2d(
                    192,
                    (1, 1),
                    padding='same',
                    name='inception_4a/1x1'
                )(pool3_3x3_s2),
                _conv2d(208, (3, 3), padding='valid', name='inception_4a/3x3')(
                    ZeroPadding2D(padding=(1, 1))(_conv2d(
                        96,
                        (1, 1),
                        padding='same',
                        name='inception_4a/3x3_reduce'
                    )(pool3_3x3_s2))),
                _conv2d(48, (5, 5), padding='valid', name='inception_4a/5x5')(
                    ZeroPadding2D(padding=(2, 2))(_conv2d(
                        16,
                        (1, 1),
                        padding='same',
                        name='inception_4a/5x5_reduce'
                    )(pool3_3x3_s2))),
                _conv2d(
                    64,
                    (1, 1),
                    padding='same',
                    name='inception_4a/pool_proj'
                )(MaxPooling2D(
                    pool_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    name='inception_4a/pool'
                )(pool3_3x3_s2))
            ]
        )

        inception_4b_output = Concatenate(axis=3, name='inception_4b/output')(
            [
                _conv2d(160, (1, 1), padding='same', name='inception_4b/1x1')(inception_4a_output),
                _conv2d(
                    224,
                    (3, 3),
                    padding='valid',
                    name='inception_4b/3x3'
                )(ZeroPadding2D(padding=(1, 1))(_conv2d(
                    112,
                    (1, 1),
                    padding='same',
                    name='inception_4b/3x3_reduce'
                )(inception_4a_output))),
                _conv2d(
                    64,
                    (5, 5),
                    padding='valid',
                    name='inception_4b/5x5'
                )(ZeroPadding2D(padding=(2, 2))(_conv2d(
                    24,
                    (1, 1),
                    padding='same',
                    name='inception_4b/5x5_reduce'
                )(inception_4a_output))),
                _conv2d(
                    64,
                    (1, 1),
                    padding='same',
                    name='inception_4b/pool_proj'
                )(MaxPooling2D(
                    pool_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    name='inception_4b/pool'
                )(inception_4a_output))
            ]
        )

        inception_4c_output = Concatenate(
            axis=3,
            name='inception_4c/output'
        )([
            _conv2d(
                128,
                (1, 1),
                padding='same',
                name='inception_4c/1x1'
            )(inception_4b_output),
            _conv2d(
                256,
                (3, 3),
                padding='valid',
                name='inception_4c/3x3'
            )(ZeroPadding2D(
                padding=(1, 1)
            )(_conv2d(
                128,
                (1, 1),
                padding='same',
                name='inception_4c/3x3_reduce'
            )(inception_4b_output))),
            _conv2d(64, (5, 5), padding='valid', name='inception_4c/5x5')(
                ZeroPadding2D(padding=(2, 2))(_conv2d(
                    24,
                    (1, 1),
                    padding='same',
                    name='inception_4c/5x5_reduce'
                )(inception_4b_output))),
            _conv2d(
                64,
                (1, 1),
                padding='same',
                name='inception_4c/pool_proj'
            )(MaxPooling2D(
                pool_size=(3, 3),
                strides=(1, 1),
                padding='same',
                name='inception_4c/pool'
            )(inception_4b_output))
        ])

        inception_4d_output = Concatenate(
            axis=3,
            name='inception_4d/output'
        )([
            _conv2d(112, (1, 1), padding='same', name='inception_4d/1x1')(inception_4c_output),
            _conv2d(288, (3, 3), padding='valid', name='inception_4d/3x3')(
                ZeroPadding2D(padding=(1, 1))(_conv2d(
                    144,
                    (1, 1),
                    padding='same',
                    name='inception_4d/3x3_reduce'
                )(inception_4c_output))),
            _conv2d(64, (5, 5), padding='valid', name='inception_4d/5x5')(
                ZeroPadding2D(padding=(2, 2))(_conv2d(
                    32,
                    (1, 1),
                    padding='same',
                    name='inception_4d/5x5_reduce'
                )(inception_4c_output))),
            _conv2d(
                64,
                (1, 1),
                padding='same',
                name='inception_4d/pool_proj'
            )(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4d/pool')(
                inception_4c_output))
        ])

        pool4_3x3_s2 = MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid',
            name='pool4/3x3_s2'
        )(PoolHelper()(ZeroPadding2D(padding=(1, 1))(Concatenate(axis=3, name='inception_4e/output')(
            [
                _conv2d(256, (1, 1), padding='same', name='inception_4e/1x1')(inception_4d_output),
                _conv2d(320, (3, 3), padding='valid', name='inception_4e/3x3')(
                    ZeroPadding2D(padding=(1, 1))(_conv2d(
                        160,
                        (1, 1),
                        padding='same',
                        name='inception_4e/3x3_reduce'
                    )(inception_4d_output))),
                _conv2d(128, (5, 5), padding='valid', name='inception_4e/5x5')(
                    ZeroPadding2D(padding=(2, 2))(_conv2d(
                        32,
                        (1, 1),
                        padding='same',
                        name='inception_4e/5x5_reduce'
                    )(inception_4d_output))),
                _conv2d(
                    128,
                    (1, 1),
                    padding='same',
                    name='inception_4e/pool_proj'
                )(MaxPooling2D(
                    pool_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    name='inception_4e/pool'
                )(inception_4d_output))
            ]))))

        inception_5a_output = Concatenate(
            axis=3,
            name='inception_5a/output'
        )([
            _conv2d(256, (1, 1), padding='same', name='inception_5a/1x1')(pool4_3x3_s2),
            _conv2d(
                320,
                (3, 3),
                padding='valid',
                name='inception_5a/3x3'
            )(ZeroPadding2D(padding=(1, 1))(_conv2d(
                160,
                (1, 1),
                padding='same',
                name='inception_5a/3x3_reduce'
            )(pool4_3x3_s2))),
            _conv2d(
                128,
                (5, 5),
                padding='valid',
                name='inception_5a/5x5'
            )(ZeroPadding2D(padding=(2, 2))(_conv2d(
                32,
                (1, 1),
                padding='same',
                name='inception_5a/5x5_reduce'
            )(pool4_3x3_s2))),
            _conv2d(
                128,
                (1, 1),
                padding='same',
                name='inception_5a/pool_proj'
            )(MaxPooling2D(
                pool_size=(3, 3),
                strides=(1, 1),
                padding='same',
                name='inception_5a/pool'
            )(pool4_3x3_s2))
        ])

        return [
            _act()(_dense_c(n=1)(
                Dropout(rate=0.7)(_dense(1024, activation='relu', name='loss1/fc')(Flatten()(
                    _conv2d(
                        128,
                        (1, 1),
                        padding='same'
                        , name='loss1/conv'
                    )(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss1/ave_pool')(
                        inception_4a_output))))))),
            _act()(_dense_c(n=2)(
                Dropout(rate=0.7)(_dense(1024, activation='relu', name='loss2/fc')(Flatten()(
                    _conv2d(
                        128,
                        (1, 1),
                        padding='same',
                        name='loss2/conv'
                    )(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss2/ave_pool')(
                        inception_4d_output))))))),
            _act(name='prob')(_dense_c(n=3)(Dropout(rate=0.4)(Flatten()(
                AveragePooling2D(pool_size=(7, 7), strides=(1, 1), name='pool5/7x7_s2')(
                    Concatenate(axis=3, name='inception_5b/output')(
                        [
                            _conv2d(
                                384,
                                (1, 1),
                                padding='same',
                                name='inception_5b/1x1'
                            )(inception_5a_output),
                            _conv2d(
                                384,
                                (3, 3),
                                padding='valid',
                                name='inception_5b/3x3'
                            )(ZeroPadding2D(padding=(1, 1))(_conv2d(
                                192,
                                (1, 1),
                                padding='same',
                                name='inception_5b/3x3_reduce'
                            )(inception_5a_output))),
                            _conv2d(128, (5, 5), padding='valid', name='inception_5b/5x5', )(
                                ZeroPadding2D(padding=(2, 2))(_conv2d(
                                    48,
                                    (1, 1),
                                    padding='same',
                                    name='inception_5b/5x5_reduce'
                                )(inception_5a_output))),
                            _conv2d(
                                128,
                                (1, 1),
                                padding='same',
                                name='inception_5b/pool_proj'
                            )(MaxPooling2D(
                                pool_size=(3, 3),
                                strides=(1, 1),
                                padding='same',
                                name='inception_5b/pool'
                            )(inception_5a_output))
                        ]
                    ))))))
        ]






def gnet_layer_classes():
    from tensorflow.keras.layers import Layer
    class LRN(Layer):

        def __init__(self, alpha=0.0001, k=1, beta=0.75, n=5, **kwargs):
            self.alpha = alpha
            self.k = k
            self.beta = beta
            self.n = n
            super(LRN, self).__init__(**kwargs)

        def call(self, x, mask=None):
            from tensorflow.keras.backend import square
            from tensorflow import pad
            b, r, c, ch = x.shape
            half_n = self.n // 2  # half the local region
            input_sqr = square(x)  # square the input
            input_sqr = pad(input_sqr, [[0, 0], [0, 0], [0, 0], [half_n, half_n]])
            scale = self.k  # offset for the scale
            norm_alpha = self.alpha / self.n  # normalized alpha
            for i in range(self.n):
                scale += norm_alpha * input_sqr[:, :, :, i:i + ch]
            scale = scale**self.beta
            x = x / scale
            return x

        def get_config(self):
            config = {'alpha': self.alpha,
                      'k'    : self.k,
                      'beta' : self.beta,
                      'n'    : self.n}
            base_config = super(LRN, self).get_config()

            yes_this_makes_a_dict = dict(list(base_config.items()) + list(config.items()))

            return yes_this_makes_a_dict


    class PoolHelper(Layer):
        def __init__(self, **kwargs):
            super(PoolHelper, self).__init__(**kwargs)

        def call(self, x, mask=None):
            return x[:, 1:, 1:, :]

        def get_config(self):
            config = {}
            base_config = super(PoolHelper, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
    return LRN, PoolHelper
