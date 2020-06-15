from typing import Optional

import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.models import Model
import tensorflow as tf
from lib.nn.gen_preproc_ims import PreDataset
import lib.nn.net_mets as net_mets
import lib.nn.nnstate as nnstate
from lib.defaults import *
from lib.nn.nn_lib import RSA
from lib.nn.nnstate import reset_global_met_log
data_folder = File('/home/matt/data')
from abc import abstractmethod, ABC

class SymNet(ABC):

    @classmethod
    @abstractmethod
    def ARCH_LABEL(cls): pass



    VERBOSE_MODE = 2  # 0=silent,1=progress bar,2=one line per epoch
    INTER_LAY = -2
    def __init__(self, max_num_classes=2, batch_normalize=False, proto=False):
        self.batch_normalize = batch_normalize
        self.max_num_classes = max_num_classes
        self.startTime = log('creating DNN: ' + self.__class__.__name__)
        if proto:
            self.net = self.build_proto_model(max_num_classes)
        else:
            self.net = self.build_model()
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
        ds = self.train_data.dataset(self.HEIGHT_WIDTH())
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
        ds = self.val_data.dataset(self.HEIGHT_WIDTH())
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

    def test_record(self, ei):
        nnstate.CURRENT_PRED_MAP = self.train_data.class_label_map
        nnstate.CURRENT_TRUE_MAP = self.test_data.class_label_map
        ds = self.test_data.dataset(self.HEIGHT_WIDTH())
        steps = self.test_data.num_steps
        log('Recording(1)... (ims=$,steps=$)', len(self.test_data), steps)
        net_mets.cmat = zeros(
            len(listkeys(nnstate.CURRENT_PRED_MAP)),
            len(listkeys(nnstate.CURRENT_TRUE_MAP)))

        inter_lay_name = self.net.layers[self.INTER_LAY].name
        inter_output_model = keras.models.Model(self.net.input, self.net.get_layer(index=self.INTER_LAY).output)

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
        RSA('Inter', inter_activations, y_true, ei, layer_name=inter_lay_name, layer_i=self.INTER_LAY)
        RSA('Raw', raw_images, y_true, ei)

        for met in net_mets.METS_TO_USE():
            met(y_true, y_pred)

        log('done recording.')

    @classmethod
    @abstractmethod
    def HEIGHT_WIDTH(cls): pass

    def build_proto_model(self, num_classes):
        inputs = Input(shape=(self.HEIGHT_WIDTH(), self.HEIGHT_WIDTH(), 3))

        flat = tf.keras.layers.Flatten()(inputs)
        dense = Dense(num_classes+1)(flat)
        prediction = Activation('softmax', name='softmax')(dense)
        m = Model(inputs, prediction)
        return m

    @abstractmethod
    def build_model(self): pass

    def DenseB(self, *args, **kwargs):
        def f(inp):
            layer = Dense(*args, **kwargs)(inp)
            if self.batch_normalize:
                # noinspection PyCallingNonCallable
                layer = BatchNormalization()(layer)
            return layer

        return f

    def Conv2DB(self, *args, **kwargs):
        def f(input):
            layer = Conv2D(*args, **kwargs)(input)
            if self.batch_normalize:
                # noinspection PyCallingNonCallable
                layer = BatchNormalization(axis=3)(layer)
            return layer

        return f
