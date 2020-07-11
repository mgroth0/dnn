from tensorflow.keras import backend
from tensorflow.python.keras.utils.conv_utils import convert_kernel
from typing import Optional
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.models import Model
import tensorflow as tf
import numpy as np

from lib.nn.gen_preproc_ims import PreDataset
import lib.nn.net_mets as net_mets
import lib.nn.nnstate as nnstate
from lib.nn.nn_lib import RSA
from lib.nn.nnstate import reset_global_met_log
from mlib.boot import log
from mlib.boot.bootutil import cn
from mlib.boot.mlog import warn
from mlib.boot.mutil import listkeys, err, zeros, arr, itr
from mlib.file import File, Folder
from mlib.term import log_invokation
data_folder = File('/home/matt/data')
from abc import abstractmethod, ABC

class SymNet(ABC):
    def data_format(self):
        if self.CA == 1:
            data_format = 'channels_first'
        elif self.CA == 3:
            data_format = None
        else:
            err('bad CA')
        return data_format
    class Meta:
        def __init__(
                self, *,
                WEIGHTS,
                FLIPPED_CONV_WEIGHTS=False,
                FULL_NAME,
                CREDITS,
                ARCH_LABEL,
                HEIGHT_WIDTH,
                INTER_LAY=-2,
                CHANNEL_AXIS=3
        ):
            self.WEIGHTS = WEIGHTS
            self.FLIPPED_CONV_WEIGHTS = FLIPPED_CONV_WEIGHTS
            self.FULL_NAME = FULL_NAME
            self.CREDITS = CREDITS
            self.ARCH_LABEL = ARCH_LABEL
            self.HEIGHT_WIDTH = HEIGHT_WIDTH
            self.INTER_LAY = INTER_LAY
            self.CHANNEL_INDEX = CHANNEL_AXIS - 1
            self.CHANNEL_AXIS = CHANNEL_AXIS
            if CHANNEL_AXIS == 3:
                self.ROW_AXIS = 1
                self.COL_AXIS = 2
            elif CHANNEL_AXIS == 1:
                self.ROW_AXIS = 2
                self.COL_AXIS = 3
            else:
                err(f'bad channel axis: {CHANNEL_AXIS}')
            self.ROW_INDEX = self.ROW_AXIS - 1
            self.COL_INDEX = self.COL_AXIS - 1
    @property
    def CI(self): return self.META().CHANNEL_INDEX
    @property
    def CA(self): return self.META().CHANNEL_AXIS


    @abstractmethod
    def META(self) -> Meta: pass



    WEIGHTS_PATH = Folder('_weights')
    ONNX_WEIGHTS_PATH = WEIGHTS_PATH['matlab']
    def weightsf(self): return self.WEIGHTS_PATH[self.META().WEIGHTS].abspath
    def oweightsf(self): return self.ONNX_WEIGHTS_PATH[f'{self.META().ARCH_LABEL}.onnx'].abspath





    VERBOSE_MODE = 2  # 0=silent,1=progress bar,2=one line per epoch
    def __init__(self, max_num_classes=2, proto=False):
        self.max_num_classes = max_num_classes
        self.proto = proto
    def build(self):
        self.startTime = log('creating DNN: ' + self.__class__.__name__)
        if self.proto:
            self.net = self.build_proto_model(self.max_num_classes)
        else:
            dims = [self.META().HEIGHT_WIDTH, self.META().HEIGHT_WIDTH, self.META().HEIGHT_WIDTH]
            dims[self.CI] = 3
            self.inputs = Input(tuple(dims))

            self.net = Model(
                inputs=self.inputs,
                outputs=self.assemble_layers(),
                name=self.META().FULL_NAME.replace(' ', '_')
            )

            arch_summary_folder = Folder('_arch')
            arch_summary_folder.mkdirs()

            model_save_file = f'_arch/{self.META().ARCH_LABEL}'
            model_pretrained_save_file = f'{model_save_file}_pretrained'

            self.net.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy'
            )
            log('saving model...')
            try:
                self.net.save(model_save_file)
                self.net.save(model_save_file + '.h5')
                log('saved model')
            except TypeError:
                warn(f'cloud not save model due to tf bug')
                File(model_save_file).deleteIfExists()
                File(model_save_file + '.h5').deleteIfExists()

            if self.META().WEIGHTS is not None:
                # transfer learning

                try:
                    self.net.load_weights(self.weightsf())
                except:
                    import traceback
                    print(traceback.format_exc())
                    ww = File(self.weightsf()).load()  # DEBUG
                    for k in listkeys(ww):
                        for kk in listkeys(ww[k]):
                            print(f'{kk}: {ww[k][kk].shape}')
                    breakpoint()
                    # 'try: self.net.load_weights(self.weightsf())\nexcept: traceback.print_exc()'

                self.net.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy'
                )
                log('saving model...')
                try:
                    self.net.save(model_pretrained_save_file)
                    self.net.save(model_pretrained_save_file + '.h5')
                    log('saved model')
                except TypeError:
                    warn(f'cloud not save model due to tf bug')
                    File(model_pretrained_save_file).deleteIfExists()
                    File(model_pretrained_save_file + '.h5').deleteIfExists()

                import h5py
                weights_file = h5py.File(self.weightsf(), "r")
                weights_report_file = arch_summary_folder[
                    f'{self.META().ARCH_LABEL}_weights.txt'
                ]
                o_weights_report_file = arch_summary_folder[
                    f'{self.META().ARCH_LABEL}_weights_matlab.txt'
                ]
                weights_report_file.write('')

                def processGroup(group, rep, indent=0):
                    for k in listkeys(group):
                        rep += '\t' * indent
                        rep += k
                        item = group[k]
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

            # Theano > Tensorflow, just flips the weight arrays in the first 2 dims. Doesn't change shape.
            if self.META().FLIPPED_CONV_WEIGHTS:
                for layer in self.net.layers:
                    if layer.__class__.__name__ == 'Conv2D':
                        original_w = backend.eval(layer.kernel)
                        converted_w = convert_kernel(original_w)
                        layer.kernel.assign(converted_w)

            arch_summary_file = arch_summary_folder[f'{self.META().ARCH_LABEL}.txt']
            log('writing summary')
            with open(arch_summary_file, 'w') as fh:
                self.net.summary(print_fn=lambda x: fh.write(x + '\n'))
            arch_summary_im = arch_summary_folder[f'{self.META().ARCH_LABEL}.png']
            log('plotting model')
            tf.keras.utils.plot_model(
                self.net,
                to_file=arch_summary_im.abspath,
                show_shapes=True,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=True,
                dpi=96,
            )
            log('finished plotting model')

        log('created DNN')
        log('compiling network...')
        mets_for_compile = []

        # nnstate.NUM_PRED_CLASSES = num_classes
        for m in net_mets.METS_TO_USE():
            mets_for_compile.append(m)

        self.net.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=mets_for_compile
        )
        self.net.run_eagerly = True

        log('compiled network!')

        # num_classes
        reset_global_met_log()
        self.train_data: Optional[PreDataset] = None
        self.val_data: Optional[PreDataset] = None
        self.test_data: Optional[PreDataset] = None


    def train(self):
        log('training network...')
        nnstate.CURRENT_PRED_MAP = self.train_data.class_label_map
        nnstate.CURRENT_TRUE_MAP = self.train_data.class_label_map
        ds = self.train_data.dataset(self.META().HEIGHT_WIDTH)
        steps = self.train_data.num_steps
        log('Training... (ims=$,steps=$)', len(self.train_data), steps)
        net_mets.cmat = zeros(
            len(listkeys(nnstate.CURRENT_PRED_MAP)),
            len(listkeys(nnstate.CURRENT_TRUE_MAP)))
        rrr = self.net.fit(
            # x,y,
            ds,
            epochs=1,
            verbose=self.VERBOSE_MODE,
            use_multiprocessing=True,
            workers=16,
            steps_per_epoch=steps,
            shuffle=False
        )

        return rrr

    def val_eval(self):
        nnstate.CURRENT_TRUE_MAP = self.val_data.class_label_map
        ds = self.val_data.dataset(self.META().HEIGHT_WIDTH)
        steps = self.val_data.num_steps
        log('Testing... (ims=$,steps=$)', len(self.val_data), steps)
        net_mets.cmat = zeros(
            len(listkeys(nnstate.CURRENT_PRED_MAP)),
            len(listkeys(nnstate.CURRENT_TRUE_MAP)))

        nnstate.TEST_STEPS = steps
        return self.net.evaluate(
            ds
            , verbose=self.VERBOSE_MODE
            , steps=steps,
            use_multiprocessing=True,
            workers=16,
        )

    @log_invokation
    def test_record(self, ei):
        nnstate.CURRENT_PRED_MAP = self.train_data.class_label_map
        nnstate.CURRENT_TRUE_MAP = self.test_data.class_label_map
        ds = self.test_data.dataset(self.META().HEIGHT_WIDTH)
        steps = self.test_data.num_steps
        log('Recording(1)... (ims=$,steps=$)', len(self.test_data), steps)
        net_mets.cmat = zeros(
            len(listkeys(nnstate.CURRENT_PRED_MAP)),
            len(listkeys(nnstate.CURRENT_TRUE_MAP)))

        inter_lay_name = self.net.layers[self.META().INTER_LAY].name
        inter_output_model = keras.models.Model(self.net.input, self.net.get_layer(index=self.META().INTER_LAY).output)

        y_pred = arr(self.net.predict(
            ds,
            steps=steps,
            verbose=1,
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
            verbose=1,
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

        RSA('Output', y_pred, y_true, ei, layer_name='Output', layer_i='-1')
        RSA('Inter', inter_activations, y_true, ei, layer_name=inter_lay_name, layer_i=self.META().INTER_LAY)
        RSA('Raw', raw_images, y_true, ei)

        for met in net_mets.METS_TO_USE():
            met(y_true, y_pred)

        log('done recording.')

    def build_proto_model(self, num_classes):
        err('get CA and CI right')
        inputs = Input(shape=(self.META().HEIGHT_WIDTH, self.META().HEIGHT_WIDTH, 3))
        flat_layer = tf.keras.layers.Flatten()(inputs)
        dense = Dense(num_classes + 1)(flat_layer)
        prediction = Activation('softmax', name='softmax')(dense)
        m = Model(inputs, prediction)
        return m

    @abstractmethod
    def assemble_layers(self): pass
