from time import time

from arch import INC
from arch.proko_inc import CustomInceptionResNetV2
from files import SALIENCE_RESULT_FOLDER
from lib.misc.scripts.asd_to_recycle_lib import BATCH_SIZE, proko_train, tf
from lib.misc.scripts.keras_models_to_test import models_to_test
from mlib.JsonSerializable import obj

NUM_EPOCHS = 1

experiment = obj([{
    'filekey'         : '',
    'models'          : {'customINC': lambda: CustomInceptionResNetV2},
    'num_ims'         : range(1, 2, 1),
    'loss'            : None,
    'preprocess_class': tf.keras.applications.inception_resnet_v2,
    'classes'         : None
}, {
    'filekey'         : '_zoo',
    'models'          : models_to_test,
    'num_ims'         : [BATCH_SIZE],
    'loss'            : 'binary_crossentropy',
    'preprocess_class': None,
    'classes'         : 2
}][0])

data_result = []
my_result_fold = SALIENCE_RESULT_FOLDER[f'keras{experiment.filekey}_{int(time())}'].mkdirs()

for name, model in list(experiment.models_to_test.items()):
    model_class = model()
    for i in experiment.num_ims:
        data_result.append({
            'model_name': name,
            'num_images': i,
            'history'   : proko_train(
                model_class,
                NUM_EPOCHS,
                i,
                HEIGHT_WIDTH=INC.HEIGHT_WIDTH,
                preprocess_class=experiment.preprocess_class,
                **({'loss': experiment.loss} if experiment.loss is not None else {}),
                **({'classes': experiment.classes} if experiment.classes is not None else {})
            ).history
        })
        my_result_fold['data_result.json'].save(data_result)
