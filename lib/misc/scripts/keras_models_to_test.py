# from tensorflow.python.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from tensorflow.python.keras.applications.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.applications.nasnet import NASNetLarge  # , # NASNetMobile
from tensorflow.python.keras.applications.resnet import ResNet101, ResNet152, ResNet50
from tensorflow.python.keras.applications.resnet_v2 import ResNet101V2
# from tensorflow.python.keras.applications.vgg16 import VGG16
# from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.xception import Xception
models_to_test = {
    'Xception'         : lambda: Xception,




    # For these valueerrors, I think that they didn't occur in the tensorflow-latest(2.4?) image but are occuring in mine(2.2?) simply because of an old bug! https://github.com/tensorflow/tensorflow/issues/41537
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