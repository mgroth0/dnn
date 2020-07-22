from arch.assembled_model import AssembledModel

class PROTO(AssembledModel):
    # def META(self): return self.Meta(
    WEIGHTS = None
    FULL_NAME = 'Prototype'
    CREDITS = 'me'
    ARCH_LABEL = 'PROTO'
    HEIGHT_WIDTH = 250
    # )
    def assemble_layers(self):
        from tensorflow.python.keras.layers import Activation, Dense
        import tensorflow as tf
        return Activation('softmax', name='softmax')(Dense(2 + 1)(tf.keras.layers.Flatten()(self.inputs)))
