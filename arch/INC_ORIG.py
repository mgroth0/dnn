from arch.pre_assembled_model import PreAssembledModel

INC_HW = 299

class INC_ORIG(PreAssembledModel):
    #     STATIC = PreAssembledModel.STATIC_ATTS(
    FULL_NAME = 'Inception Resnet V2 (original)'
    IS_PRETRAINED = True
    CREDITS = 'Keras Applications'
    ARCH_LABEL = 'INC_ORIG'
    HEIGHT_WIDTH = INC_HW
    # )

    def build_net(self):
        import tensorflow as tf
        self.net = tf.keras.applications.InceptionResNetV2(
            include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
            pooling=None, classes=1000, classifier_activation='softmax'
        )