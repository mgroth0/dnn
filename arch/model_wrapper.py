print('model_wrapper.py: top')
from copy import deepcopy

import time
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
print('model_wrapper.py: finished imports')

_SAVE_MODEL = False
_PLOT_MODEL = False

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
    def build(self,FLAGS):
        self.build_net(FLAGS)
        assert isinstance(self.net, self.tf.keras.Model)
        self.write_arch_summary()
        if _PLOT_MODEL:
            self.plot_model()
        if _SAVE_MODEL:
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


    def predict(self, inputs, verbose=1, **kwargs) -> np.array:
        # print('model_wrapper.py: importing tf')
        import tensorflow as tf
        # print('model_wrapper.py: finished importing tf')
        if not isinstance(inputs, types.GeneratorType) and not isinstance(inputs, tf.keras.utils.Sequence):
            if len(inputs.shape) == 3:
                inputs = np.expand_dims(inputs, axis=0)
        y_pred = self.net.predict(
            inputs, verbose=verbose, **kwargs
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



def simple_predict(net: ModelWrapper, pp, inputs, *, length):
    # vs_n = [(V_Stacker(), n) for n in nets]
    import tensorflow as tf
    class Timer:
        def __init__(self, name):
            self._tic = None
            self.name = name
            self.disabled = False
        def tic(self):
            if not self.disabled: self._tic = time.monotonic_ns()
        def toc(self, n):
            if not self.disabled:
                t = time.monotonic_ns() - self._tic
                log(f'{self.name}\t{n}\t{t}')
    class Gen(tf.keras.utils.Sequence):
        def __init__(self):
            # self.g = self.gen()
            self.counter = 0
            self.cache = {}
        def __getitem__(self, index):
            im = inputs(index)
            # t.toc(2)
            img = pp.preprocess(im)
            # t.toc(3)
            if net.CHANNEL_AXIS == 1:
                rimg = deepcopy(img)
                # t.toc(4)
                rimg = np.swapaxes(rimg, 0, 2)
                # t.toc(5)
                # prog.tick()
                r = np.expand_dims(rimg, axis=0),
                # t.toc(6)
            else:
                # prog.tick()
                r = np.expand_dims(img, axis=0),
                # t.toc(7)
            # if i % 100 == 0 or i > 49000:
            #     log(f'finished {i} out of {len(self)}')
            # t.toc(8)
            # STATUS_FILE.write(dict(
            #     finished=i,
            #     total=len(self)
            # ))
            # t.toc(9)
            return r
            # log(f'trying to get index {index}')
            # log(f'current indices range from {safemin(list(self.cache.keys()))} to {safemax(list(self.cache.keys()))}')
            # if index in self.cache:
            #     r = self.cache[index]
            #     del self.cache[index]
            #     return r
            # else:
            #     self.cache[self.counter] = next(self.g)
            #     self.counter += 1
            #     return self[index]
        def __len__(self):
            return length
        # def gen(self):
        #     # with Progress(len(self)) as prog:
        #     # STATUS_FILE = File('status.json')
        #     # log(f'{len(inputs)=}')
        #     t = Timer('simple_predict')
        #     t.disabled = True
        #     t.tic()
        #     t.toc(1)
        #     for i, im in enum(inputs):
        #         if i == 0:
        #             log('generating first input...')
        #         t.toc(2)
        #         img = pp.preprocess(im)
        #         t.toc(3)
        #         if net.CHANNEL_AXIS == 1:
        #             rimg = deepcopy(img)
        #             t.toc(4)
        #             rimg = np.swapaxes(rimg, 0, 2)
        #             t.toc(5)
        #             # prog.tick()
        #             r = np.expand_dims(rimg, axis=0),
        #             t.toc(6)
        #         else:
        #             # prog.tick()
        #             r = np.expand_dims(img, axis=0),
        #             t.toc(7)
        #         if i % 100 == 0 or i > 49000:
        #             log(f'finished {i} out of {len(self)}')
        #             t.toc(8)
        #             # STATUS_FILE.write(dict(
        #             #     finished=i,
        #             #     total=len(self)
        #             # ))
        #         t.toc(9)
        #         yield r
        #         t.toc(10)


    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    return net.predict(Gen(), verbose=0)  # use_multiprocessing=False, workers=16

def chain_predict(nets, pp, inputs):
    vs_n = [(V_Stacker(), n) for n in nets]
    # def gen():
    for im in inputs:
        img = pp.preprocess(im)
        for vs, n in vs_n:
            if n.CHANNEL_AXIS == 1:
                rimg = deepcopy(img)
                rimg = np.swapaxes(rimg, 0, 2)
                yield rimg
                vs += n.predict(rimg)
            else:
                yield img
                vs += n.predict(img)
    return tuple([vs.mat for vs, n in vs_n])
    # return
