from arch.GNET import gnet_layer_classes
from arch.model_wrapper import ModelWrapper
from mlib.boot.lang import enum
from mlib.boot.mlog import err
from mlib.file import File

class LoadedModel(ModelWrapper):
    # STATIC = ModelWrapper.STATIC_ATTS(
    FULL_NAME = '??? (loaded)'
    CREDITS = '???'
    # )

    def __init__(self, label, file, hw, *args, is_pretrained, **kwargs):
        super().__init__(*args, **kwargs)
        self.ARCH_LABEL = label
        self.file = File(file)
        self.IS_PRETRAINED = is_pretrained
        self.HEIGHT_WIDTH = hw

        self.OUTPUT_IDX = None

    def build_net(self):
        import tensorflow as tf
        LRN, PoolHelper = gnet_layer_classes()
        if self.file.ext == '.h5':
            self.net = tf.keras.models.load_model(
                self.file.abspath,
                custom_objects={
                    'PoolHelper': PoolHelper,
                    'LRN'       : LRN
                }
            )
        else:
            err('')
        if len(self.net.outputs) > 1:
            found = False
            for i, o in enum(self.net.outputs):
                if 'prob' in o.name:
                    assert not found
                    self.OUTPUT_IDX = i
                    found = True
            assert found
