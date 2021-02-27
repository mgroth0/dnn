import time
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from tensorflow.python.keras.applications.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.applications.nasnet import NASNetLarge, NASNetMobile
from tensorflow.python.keras.applications.resnet import ResNet101, ResNet152, ResNet50
from tensorflow.python.keras.applications.resnet_v2 import ResNet101V2

from arch import INC
from arch.proko_inc import CustomInceptionResNetV2
from lib.misc.scripts.asd_to_recycle_lib import BATCH_SIZE, proko_train, tf
from mlib.boot.mlog import err
from mlib.file import Folder

data_result = []

fold = Folder(f'_data/result/keras_{int(time.time())}').mkdirs()

# from lib. import tf

for i in range(13, 14, 1):
    if i < BATCH_SIZE:
        err('bad')
    num_epochs = 1
    history = proko_train(
        CustomInceptionResNetV2,
        num_epochs,
        i,
        HEIGHT_WIDTH=INC.HEIGHT_WIDTH,
        preprocess_class=tf.keras.applications.inception_resnet_v2
    )  # more epochs without BN is required to get to overfit
    data_result.append({
        'num_images': i,
        'history'   : history.history
    })
    fold['data_result.json'].save(data_result)

data_result = []
fold = Folder(f'_data/result/keras_zoo_{int(time.time())}').mkdirs()

# NUM_CLASSES = 1000 #2

models_to_test = {
    'Xception'         : lambda: Xception,
    # 'VGG16'            : lambda: VGG16, # ValueError: Input 0 of layer fc1 is incompatible with the layer: expected axis -1 of input shape to have value 25088 but received input with shape [12, 41472]
    # 'VGG19'            : lambda: VGG19, # ValueError: Input 0 of layer fc1 is incompatible with the layer: expected axis -1 of input shape to have value 25088 but received input with shape [12, 41472]


    'ResNet50'         : lambda: ResNet50,
    'ResNet101'        : lambda: ResNet101,
    'ResNet152'        : lambda: ResNet152,
    # 'ResNet50V3'       : lambda: ResNet50V3(classes=NUM_CLASSES),
    'ResNet101V2'      : lambda: ResNet101V2,
    # 'ResNet152V3'      : lambda: ResNet152V3,
    'InceptionV3'      : lambda: InceptionV3,
    'InceptionResNetV2': lambda: InceptionResNetV2,
    'MobileNet'        : lambda: MobileNet,
    'MobileNetV2'      : lambda: MobileNetV2,
    # 'DenseNet121'      : lambda: DenseNet121, # DenseNet121() got an unexpected keyword argument 'classifier_activation
    # 'DenseNet169'      : lambda: DenseNet169, # [MP|503.47|shell         ] TypeError: DenseNet169() got an unexpected keyword argument 'classifier_activation'

    # 'DenseNet201'      : lambda: DenseNet201, # TypeError: DenseNet201() got an unexpected keyword argument 'classifier_activation'
    # 'NASNetMobile'     : lambda: NASNetMobile, #  TypeError: NASNetMobile() got an unexpected keyword argument 'classifier_activation'
    'NASNetLarge'      : lambda: NASNetLarge,
    'EfficientNetB0'   : lambda: EfficientNetB0,
    'EfficientNetB1'   : lambda: EfficientNetB1,
    'EfficientNetB2'   : lambda: EfficientNetB2,
    'EfficientNetB3'   : lambda: EfficientNetB3,
    'EfficientNetB4'   : lambda: EfficientNetB4,
    'EfficientNetB5'   : lambda: EfficientNetB5,
    'EfficientNetB6'   : lambda: EfficientNetB6,
    'EfficientNetB7'   : lambda: EfficientNetB7,
}

for name, model in list(models_to_test.items()):
    print(f'TESTING MODEL: {name}')
    num_epochs = 1
    num_ims = 10
    model_class = model()
    history = proko_train(
        model_class,
        num_epochs,
        num_ims,
        include_top=True,  # THIS WAS THE BUG!!!! Probably used a different loss function while it was false
        # weights='imagenet',
        weights=None,
        preprocess_class=None,
        # classes=1000,
        classes=2,
        # classes = 1, #classifer activation not keyword
        # loss='categorical_crossentropy',
        # loss='mse',
        loss='binary_crossentropy'
    )  # more epochs without BN is required to get to overfit
    data_result.append({
        'model_name': name,
        'num_images': num_ims,
        'history'   : history.history
    })
    fold['data_result.json'].save(data_result)



# [MP|260.39|shell         ] tensorflow.python.framework.errors_impl.ResourceExhaustedError:  OOM when allocating tensor with shape[12,1024,19,19] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc