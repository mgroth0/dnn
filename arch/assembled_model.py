from lib.dnn_data_saving import save_dnn_data
from lib.nn.nn_lib import rsa_norm
from lib.nn.tf_lib import Verbose
from mlib.datamodel.DataModelBase import Class, ClassSet, FeatureMatrix
print('assembled_model.py: top')
from typing import Optional
import numpy as np
from abc import abstractmethod, ABC

from arch.model_wrapper import ModelWrapper
import lib.nn.net_mets as net_mets
import lib.nn.nnstate as nnstate
print('assembled_model.py: half through imports')
from mlib.abstract_attributes import Abstract
from mlib.boot import log
from mlib.boot.lang import enum, listkeys, cn
from mlib.boot.mlog import warn, err
from mlib.boot.stream import zeros, arr, itr
from mlib.file import File, Folder
from mlib.term import log_invokation
print('assembled_model.py: finished imports')



class AssembledModel(ModelWrapper, ABC):
    # @dataclass
    # class STATIC_ATTS(ModelWrapper.STATIC_ATTS):
    ARCH_LABEL = Abstract(str)
    HEIGHT_WIDTH = Abstract(int)
    WEIGHTS = Abstract(Optional[str])
    FLIPPED_CONV_WEIGHTS = False
    OUTPUT_IDX = None
    PP = Abstract(str)
    # IS_PRETRAINED: bool = field(init=False)

    @classmethod
    def __meta_post_init__(cls):
        # def __post_init__(self):
        super().__meta_post_init__()
        cls.IS_PRETRAINED = cls.WEIGHTS is not None

    WEIGHTS_PATH = Folder('_weights')
    ONNX_WEIGHTS_PATH = WEIGHTS_PATH['matlab']
    def weightsf(self): return self.WEIGHTS_PATH[self.WEIGHTS].abspath
    def oweightsf(self): return self.ONNX_WEIGHTS_PATH[f'{self.ARCH_LABEL}.onnx'].abspath


    @log_invokation(with_class=True)
    def build_net(self, FLAGS):
        dims = [self.HEIGHT_WIDTH, self.HEIGHT_WIDTH, self.HEIGHT_WIDTH]
        dims[self.CI] = 3
        from tensorflow.python.keras import Input
        self.inputs = Input(tuple(dims))
        self.net = self.tf.python.keras.models.Model(
            inputs=self.inputs,
            outputs=self.assemble_layers(),
            name=self.FULL_NAME.replace(' ', '_')
        )
        if self.WEIGHTS is not None and FLAGS.TRANSFER_LEARNING:  # transfer learning
            # breakpoint()
            self._load_weights()
            self.write_weight_reports()
            if self.FLIPPED_CONV_WEIGHTS: self._flip_conv_weights()
        elif (self.WEIGHTS is not None) and (not FLAGS.TRANSFER_LEARNING):
            log('not loading weights because TRANSFER_LEARNING is disabled')
        self._compile(net_mets.METS_TO_USE())



    @log_invokation
    def test_record(self, ei):
        nnstate.CURRENT_PRED_MAP = self.train_data.class_label_map
        nnstate.CURRENT_TRUE_MAP = self.test_data.class_label_map
        ds = self.test_data.dataset(self.HEIGHT_WIDTH)
        steps = self.test_data.num_steps
        log('Recording(1)... (ims=$,steps=$)', len(self.test_data), steps)
        net_mets.cmat = zeros(
            len(listkeys(nnstate.CURRENT_PRED_MAP)),
            len(listkeys(nnstate.CURRENT_TRUE_MAP)))

        inter_lay_name = self.net.layers[self.INTER_LAY].name
        inter_output_model = self.tf.python.keras.models.Model(self.net.input,
                                                               self.net.get_layer(index=self.INTER_LAY).output)

        y_pred = arr(self.net.predict(
            ds,
            steps=steps,
            verbose=Verbose.PROGRESS_BAR,
            use_multiprocessing=True,
            workers=16,
        ))

        log('done recording(1)')

        if len(y_pred.shape) == 3:  # GNET has 3 outputs, all identical I guess but not sure
            y_pred = y_pred[2]

        log('Recording(2)... (ims=$,steps=$)', len(self.test_data), steps)

        inter_activations = arr(inter_output_model.predict(
            ds,
            steps=steps,
            verbose=Verbose.PROGRESS_BAR,
            use_multiprocessing=True,
            workers=16
        ))

        log('done recording(2)')

        x, _ = self.test_data.x(self)
        y = self.test_data.y(self)
        y_true = arr(y).flatten()

        raw_images = x
        raw_images2 = []
        if len(x.shape) == 5:
            for batch in raw_images:
                for im in batch:
                    raw_images2.append(im)
        else:
            raw_images2 = raw_images
        raw_images = arr(raw_images2)
        raw_images2 = []
        for i in itr(raw_images):
            raw_images2.append(raw_images[i].flatten())
        raw_images = arr(raw_images2)

        inter_shape = inter_activations.shape
        inter_activations = np.reshape(inter_activations, (inter_shape[0], -1))

        BLOCK_LEN = 10  # I'm writing this bc I think it was always 10 back when I ran this code
        TEST_CLASS_MAP = nnstate.CURRENT_TRUE_MAP
        clas_set = ClassSet([Class(name=k, index=v) for k, v in TEST_CLASS_MAP.items()])
        def run_and_save_rsa(nam, mat1, layer_name=None, layer_i=None):
            index_to_cn = {v: k for k, v in TEST_CLASS_MAP.items()}
            feature_matrix = FeatureMatrix(mat1, clas_set, [Class(index_to_cn[iii], iii) for iii, yt in enum(y_true)])
            feature_matrix.sort_by_class_name()
            fd = feature_matrix.compare(rsa_norm).image_plot()
            tit = f'L2-{nam}'
            fd.title = f'{tit} ({nnstate.FLAGS.arch}{nnstate.FLAGS.ntrain}E{ei + 1})'
            if nam == 'Inter':
                fd.title = f'{fd.title}(Layer{layer_i}:{layer_name})'
            save_dnn_data(fd, tit, f'CM{ei + 1}', 'mfig')

        run_and_save_rsa('Output', y_pred, layer_name='Output', layer_i='-1')
        run_and_save_rsa('Inter', inter_activations, layer_name=inter_lay_name, layer_i=self.INTER_LAY)
        run_and_save_rsa('Raw', raw_images)

        for met in net_mets.METS_TO_USE():
            met(y_true, y_pred)

        log('done recording.')

    @abstractmethod
    def assemble_layers(self): pass






    def _load_weights(self):
        try:
            self.net.load_weights(self.weightsf())
        except:
            import traceback
            print(traceback.format_exc())
            ww = File(self.weightsf()).load()  # DEBUG
            for k in listkeys(ww):
                for kk in listkeys(ww[k]):
                    print(f'{kk}: {ww[k][kk].shape}')
            err('could not load weights')

    def write_weight_reports(self):
        import h5py
        weights_file = h5py.File(self.weightsf(), "r")
        weights_report_file = self.arch_summary_folder[
            f'{self.ARCH_LABEL}_weights.txt'
        ]
        o_weights_report_file = self.arch_summary_folder[
            f'{self.ARCH_LABEL}_weights_matlab.txt'
        ]
        weights_report_file.write('')

        def processGroup(group, rep, indent=0):
            for ke in listkeys(group):
                rep += '\t' * indent
                rep += ke
                item = group[ke]
                if 'Dataset' in cn(item):
                    # c = 'Dataset'
                    rep += f'\t\t{item.shape} {item.dtype}\n'
                elif 'Group' in cn(item):
                    # c = 'Group'
                    rep += '\n'
                    rep = processGroup(item, rep, indent + 1)
                    # sub = f'{item.shape} {item.dtype}'
                else:
                    err(f'what is this: {cn(item)}')
            return rep

        report = ''
        report = processGroup(weights_file, report)
        log('writing weights report...')
        weights_report_file.write(report)
        log('finished writing weights report')

        log('writing matlab weight report...')
        warn('THERE ARE 2 VERSIONS OF THE ONNX FILES IN _weights/matlab AND I DONT KNOW THE DIFFERENCE')
        import onnx
        o_model = onnx.load(self.oweightsf())
        o_weights_report_file.write(repr(o_model.graph.node))
        log('finished writing matlab weight report...')

    def _flip_conv_weights(self):
        # Theano > Tensorflow, just flips the weight arrays in the first 2 dims. Doesn't change shape.
        for layer in self.net.layers:
            if layer.__class__.__name__ == 'Conv2D':
                from tensorflow.keras import backend
                original_w = backend.eval(layer.kernel)
                from tensorflow.python.keras.utils.conv_utils import convert_kernel
                converted_w = convert_kernel(original_w)
                layer.kernel.assign(converted_w)
