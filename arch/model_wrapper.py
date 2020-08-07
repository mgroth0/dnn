from copy import deepcopy
import types

from typing import Optional
from abc import abstractmethod, ABC
import numpy as np

from lib.nn.gen_preproc_ims import PreDataset
from mlib.abstract_attributes import AbstractAttributes, Abstract
from mlib.boot.mlog import err, log, warn
from mlib.boot.stream import V_Stacker
from mlib.file import Folder, File
from mlib.term import log_invokation
import tensorflow as tf
def simple_predict(net,pp,inputs,*,length):
    # vs_n = [(V_Stacker(), n) for n in nets]
    class Gen(tf.keras.utils.Sequence):
        def __init__(self):
            self.g = self.gen()
        def __getitem__(self, index):
            return next(self.g)
        def __len__(self):
            return length
        def gen(self):
            for im in inputs:
                img = pp.preprocess(im)
                if net.CHANNEL_AXIS == 1:
                    rimg = deepcopy(img)
                    try:
                        rimg = np.swapaxes(rimg, 0, 2)
                    except:
                        breakpoint()
                    log('yeilding')
                    yield np.expand_dims(rimg, axis=0),
                else:
                    log('yeilding')
                    yield np.expand_dims(img, axis=0),
    return net.predict(Gen())

def chain_predict(nets, pp, inputs):
    vs_n = [(V_Stacker(), n) for n in nets]
    # def gen():
    for im in inputs:
        img = pp.preprocess(im)
        for vs, n in vs_n:
            if n.CHANNEL_AXIS == 1:
                rimg = deepcopy(img)
                try:
                    rimg = np.swapaxes(rimg, 0, 2)
                except:
                    breakpoint()
                yield rimg
                vs += n.predict(rimg)
            else:
                yield img
                vs += n.predict(img)
    return tuple([vs.mat for vs, n in vs_n])
    # return

class ModelWrapper(AbstractAttributes, ABC):
    IMAGE_NET_FOLD = Folder('_ImageNetTesting')

    # @dataclass
    # class STATIC_ATTS(STATIC_ATTS):
    FULL_NAME = Abstract(str)
    CREDITS = Abstract(str)
    INTER_LAY = -2
    CHANNEL_AXIS = 3

    @classmethod
    def __meta_post_init__(cls):
        # ROW_AXIS: int = field(init=False)
        # COL_AXIS: int = field(init=False)
        # ROW_INDEX: int = field(init=False)
        # COL_INDEX: int = field(init=False)
        # CHANNEL_INDEX: int = field(init=False)

        # def __post_init__(self):
        assert cls.CHANNEL_AXIS in [1, 3]
        cls.ROW_AXIS = 1 if cls.CHANNEL_AXIS == 3 else 2
        cls.COL_AXIS = 2 if cls.CHANNEL_AXIS == 3 else 3
        cls.ROW_INDEX = cls.ROW_AXIS - 1
        cls.COL_INDEX = cls.COL_AXIS - 1
        cls.CHANNEL_INDEX = cls.CHANNEL_AXIS - 1



    def __init__(self, *args, **kwargs):
        super().__init__()
        import tensorflow as tf
        self.tf = tf
        self.net = None
        self.train_data: Optional[PreDataset] = None
        self.val_data: Optional[PreDataset] = None
        self.test_data: Optional[PreDataset] = None

    @abstractmethod
    def build_net(self): pass

    @log_invokation
    def build(self):
        self.build_net()
        assert isinstance(self.net, self.tf.keras.Model)
        self.write_arch_summary()
        self.plot_model()
        self._save(pretrained=self.pretrained)
        self.net.run_eagerly = True
        return self


    @property
    def CI(self): return self.CHANNEL_INDEX
    @property
    def CA(self): return self.CHANNEL_AXIS


    def data_format(self):
        if self.CA == 1:
            return 'channels_first'
        elif self.CA == 3:
            return None
        else:
            err('bad CA')

    arch_summary_folder = Folder('_arch')
    arch_summary_folder.mkdirs()
    def write_arch_summary(self):
        arch_summary_file = self.arch_summary_folder[f'{self.ARCH_LABEL}.txt']
        log('writing summary')
        with open(arch_summary_file, 'w') as fh:
            self.net.summary(print_fn=lambda x: fh.write(x + '\n'))
    @log_invokation()
    def plot_model(self):
        arch_summary_im = self.arch_summary_folder[f'{self.ARCH_LABEL}.png']
        try:
            self.tf.keras.utils.plot_model(
                self.net,
                to_file=arch_summary_im.abspath,
                show_shapes=True,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=True,
                dpi=96,
            )
        except AssertionError as e:
            # I think there are sometimes problems creating InceptionResNetV2s plot. This makes sense considering that it is huge. I think AssertionError is thrown when its too big but I'm not sure
            arch_summary_im.deleteIfExists()
            arch_summary_im.res_pre_ext('_sorry').resrepext('txt').write(f'{repr(e)}')
    @log_invokation()
    def _save(self, pretrained=False):
        model_save_file = f'_arch/{self.ARCH_LABEL}'
        if pretrained:
            model_save_file = f'{model_save_file}_pretrained'
        try:
            self.net.save(model_save_file)
            self.net.save(f'{model_save_file}.h5')
            log('saved model')
        except TypeError:
            warn(f'could not save model due to tf bug')
            File(model_save_file).deleteIfExists()
            File(f'{model_save_file}.h5').deleteIfExists()


    @property
    def pretrained(self): return self.IS_PRETRAINED
    @property
    def hw(self): return self.HEIGHT_WIDTH
    @property
    def label(self): return self.ARCH_LABEL


    def predict(self, inputs) -> np.array:
        if not isinstance(inputs, types.GeneratorType) and not isinstance(inputs,tf.keras.utils.Sequence):
            if len(inputs.shape) == 3:
                inputs = np.expand_dims(inputs, axis=0)
        y_pred = self.net.predict(
            inputs, verbose=1,
        )
        if self.OUTPUT_IDX is not None:
            y_pred = y_pred[self.OUTPUT_IDX]
        return y_pred

    def from_ML_vers(self):
        from arch.loaded_model import LoadedModel
        return LoadedModel(
            self.label.replace('_ORIG', ''),
            f'_data/darius_pretrained/{self.label.replace("_ORIG", "")}_pretrained.onnx',
            self.hw,
            is_pretrained=True
        )
