import json

import tensorflow as tf

from mlib.boot import log
from mlib.boot.lang import enum
from mlib.boot.stream import listitems
from mlib.file import File, Folder
from mlib.str import utf_decode
def count():
    log('count here 1')
    data = {
        'train'     : count_split("train"),
        'validation': count_split("validation"),
    }

    real_data = {}
    for k, v in listitems(data['train']):
        real_data[k] = {'train': v}
    for k, v in listitems(data['validation']):
        real_data[k]['validation'] = v

    real_data = json.dumps(real_data,indent=2)
    log(f'data sample: {real_data[:20]}')

    File('imagenet_count.json').write(real_data)

def count_split(spl):
    data = {}
    root = Folder('/matt/data/ImageNet/output_tf')
    filenames = root.glob(f'{spl}*').map(lambda x: x.abspath).tolist()  # validation
    ds = tf.data.TFRecordDataset(filenames)

    image_feature_description = {
        'image/height'      : tf.io.FixedLenFeature([], tf.int64),
        'image/width'       : tf.io.FixedLenFeature([], tf.int64),
        'image/colorspace'  : tf.io.FixedLenFeature([], tf.string),
        'image/channels'    : tf.io.FixedLenFeature([], tf.int64),
        'image/class/label' : tf.io.FixedLenFeature([], tf.int64),
        'image/class/synset': tf.io.FixedLenFeature([], tf.string),
        'image/class/text'  : tf.io.FixedLenFeature([], tf.string),
        'image/format'      : tf.io.FixedLenFeature([], tf.string),
        'image/filename'    : tf.io.FixedLenFeature([], tf.string),
        'image/encoded'     : tf.io.FixedLenFeature([], tf.string),
    }
    log('looping imagenet')

    for i, raw_record in enum(ds):
        example = tf.io.parse_single_example(raw_record, image_feature_description)
        if i % 100 == 0:
            log(f'on image {i}')
        classname = utf_decode(example['image/class/text'].numpy())
        if classname not in data:
            data[classname] = 1
        else:
            data[classname] += 1
    return data
