from lib.datamodel.Classification import Class, ClassSet
from mlib.boot.lang import enum
RSA_CLASSES = ClassSet([Class(name=n, index=i) for i, n in enum([
    'NS0',
    'NS2',
    'NS4',
    'NS6',
    'NSd4',
    'S0',
    'S2',
    'S4',
    'S6',
    'Sd4'
])])



RSA_LAYERS = {
    "SQN"      : 'relu_conv10',  # 784
    "AlexNet"  : 'fc7',  # 4096
    "GoogleNet": 'inception_5b-output',  # 50176
    "IRN"      : 'conv_7b_ac',  # 98304
    "IV3"      : 'mixed10',  # 131072
    "RN18"     : 'res5b-relu',  # 25088,
    "LSTM"     : 'final cell'
}