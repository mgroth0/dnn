from arch.GNET import gnet_layer_classes
from arch.model_wrapper import ModelWrapper
from mlib.boot.lang import enum, ismac, HOME
from mlib.boot.mlog import err
from mlib.file import File
from mlib.shell import eshell

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

    def build_net(self,FLAGS):
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
        elif self.file.ext == 'onnx':
            if ismac():
                onnx_tf = f'{HOME}/miniconda3/envs/dnn/bin/onnx-tf'
            else:
                onnx_tf = f'matt/miniconda3/envs/dnn/bin/onnx-tf'
            out = self.file.res_pre_ext("pb")
            eshell(f'{onnx_tf} convert -i {self.file.abspath} -o {out.abspath}')

            # onnx-tf convert -i /path/to/input.onnx -o /path/to/output.pb
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
