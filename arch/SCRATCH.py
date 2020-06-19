from arch.shared import *
from tensorflow.keras.layers import (
    Add,
    AveragePooling2D,
    BatchNormalization,


    Flatten





)
from tensorflow.keras.regularizers import l2

from mlib.boot.mutil import functionalize


class SCRATCH(SymNet):
    def META(self): return self.Meta(
        WEIGHTS=None,

        FULL_NAME='ResNet18',
        CREDITS='https://github.com/raghakot/keras-resnet',
        ARCH_LABEL='SCRATCH',
        HEIGHT_WIDTH=299

    )
    def assemble_layers(self):
        conv1 = self._conv_bn_relu(
            filters=64,
            kernel_size=(7, 7),
            strides=(2, 2)
        )(self.inputs)
        block = MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='same'
        )(conv1)
        filters = 64
        for i, repetition_of_block_unit in enumerate([2, 2, 2, 2]):
            # At each block unit, the number of filters are doubled and the input size is halved
            block = self._residual_block(
                self.basic_block,
                filters=filters,
                repetitions=repetition_of_block_unit,
                is_first_layer=(i == 0)
            )(block)
            filters *= 2

        return Dense(
            units=self.max_num_classes,
            kernel_initializer='he_normal',
            activation='softmax'
        )(Flatten(
        )(self._average_pool(
            self._bn_relu(
                # Last activation, Classifier block
                block
            )
        )))


    def _average_pool(self, inputs):
        block_shape = tuple(inputs.shape.as_list())
        return AveragePooling2D(
            pool_size=(
                block_shape[self.META().ROW_AXIS],
                block_shape[self.META().COL_AXIS]
            ),
            strides=(1, 1)
        )(inputs)


    def _bn_relu(self, input_layer):
        """Helper to build a BN -> relu block
        """
        input_layer = BatchNormalization(axis=self.META().CHANNEL_AXIS)(input_layer)
        return Activation('relu')(input_layer)

    def _conv_bn_relu(self, filters=64, kernel_size=(7, 7), strides=(2, 2)):
        """Helper to build a conv -> BN -> relu block
        """

        def f(input_layer):
            conv = Conv2D(filters=filters, kernel_size=kernel_size,
                          strides=strides, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1.e-4))(input_layer)
            return self._bn_relu(conv)

        return f

    def _bn_relu_conv(self, *, filters, kernel_size, strides=(1, 1)):
        """Helper to build a BN -> relu -> conv block.
        This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
        """
        def f(input_layer):
            activation = self._bn_relu(input_layer)
            return Conv2D(filters=filters, kernel_size=kernel_size,
                          strides=strides, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1.e-4))(activation)

        return f

    def _shortcut(self, input_layer, residual):
        """Adds a shortcut between input_layer and residual block and merges them with "sum"
        """
        # Expand channels of shortcut to match residual.
        # Stride appropriately to match residual (width, height)
        # Should be int if network architecture is correctly configured.
        input_shape = tuple(input_layer.shape.as_list())
        residual_shape = tuple(residual.shape.as_list())
        stride_width = int(round(input_shape[self.META().ROW_AXIS] / residual_shape[self.META().ROW_AXIS]))
        stride_height = int(round(input_shape[self.META().COL_AXIS] / residual_shape[self.META().COL_AXIS]))
        equal_channels = input_shape[self.META().CHANNEL_AXIS] == residual_shape[self.META().CHANNEL_AXIS]

        shortcut = input_layer
        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = Conv2D(filters=residual_shape[self.META().CHANNEL_AXIS],
                              kernel_size=(1, 1),
                              strides=(stride_width, stride_height),
                              padding='valid',
                              kernel_initializer='he_normal',
                              kernel_regularizer=l2(0.0001))(input_layer)

        return Add()([shortcut, residual])

    @staticmethod
    def _residual_block(block_function, filters, repetitions, is_first_layer=False):
        # Builds a residual block with repeating bottleneck blocks.

        def f(input_layer):
            for i in range(repetitions):
                init_strides = (1, 1)
                if i == 0 and not is_first_layer:
                    init_strides = (2, 2)
                input_layer = block_function(
                    filters=filters,
                    init_strides=init_strides,
                    is_first_block_of_first_layer=(is_first_layer and i == 0)
                )(input_layer)
            return input_layer

        return f

    # The original paper used basic_block for layers < 50
    def basic_block(self, filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
        """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
        Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
        """

        def f(input_layer):
            if is_first_block_of_first_layer:
                # don't repeat bn->relu since we just did bn->relu->maxpool
                conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                               strides=init_strides,
                               padding='same',
                               kernel_initializer='he_normal',
                               kernel_regularizer=l2(1e-4))(input_layer)
            else:
                conv1 = self._bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                           strides=init_strides)(input_layer)

            residual = self._bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
            return self._shortcut(input_layer, residual)

        return f
