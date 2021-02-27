import cv2
import os
import random
import matplotlib.image as mpimg
from lib.nn.tf_lib import Verbose
BATCH_SIZE = 10
SANITY_SWITCH = False
SANITY_MIX = True
from lib.boot.nn_init_fun import setupTensorFlow
tf = setupTensorFlow()
class_map = {'dog': 0, 'cat': 1}
def get_data(num_ims_per_class='ALL'):

    data = '/matt/data/tf_bug1'

    train_data_cat = [data + f'/Training/cat/{x}' for x in os.listdir(data + '/Training/cat')]
    train_data_dog = [data + f'/Training/dog/{x}' for x in os.listdir(data + '/Training/dog')]
    random.shuffle(train_data_cat)
    random.shuffle(train_data_dog)
    if num_ims_per_class != 'ALL':
        train_data_cat = train_data_cat[0:num_ims_per_class]
    train_data_dog = train_data_dog[0:num_ims_per_class]
    train_data = train_data_cat + train_data_dog

    test_data_cat = [data + f'/Testing/cat/{x}' for x in os.listdir(data + '/Testing/cat')]
    test_data_dog = [data + f'/Testing/dog/{x}' for x in os.listdir(data + '/Testing/dog')]
    random.shuffle(test_data_cat)
    random.shuffle(test_data_dog)
    if num_ims_per_class != 'ALL':
        test_data_cat = test_data_cat[0:num_ims_per_class]
        test_data_dog = test_data_dog[0:num_ims_per_class]
    test_data = test_data_cat + test_data_dog

    random.shuffle(train_data)
    random.shuffle(test_data)

    if SANITY_MIX:
        mixed_data = train_data + test_data
        random.shuffle(mixed_data)
        train_data = mixed_data[:len(train_data)]
        test_data = mixed_data[len(train_data):]

    if SANITY_SWITCH:
        tmp_data = train_data
        train_data = test_data
        test_data = tmp_data
    return train_data, test_data

def get_gen(data,HEIGHT_WIDTH):
    def gen():
        pairs = []
        i = 0
        for im_file in data:
            i += 1
            if i <= BATCH_SIZE:
                pairs += [preprocess(im_file,HEIGHT_WIDTH)]
            if i == BATCH_SIZE:
                yield (
                    [pair[0] for pair in pairs],
                    [pair[1] for pair in pairs]
                )
                pairs.clear()
                i = 0
    return gen

def get_ds(data,HEIGHT_WIDTH):
    return tf.data.Dataset.from_generator(
        get_gen(data),
        (tf.float32, tf.int64),
        output_shapes=(
            tf.TensorShape((BATCH_SIZE, HEIGHT_WIDTH, HEIGHT_WIDTH, 3)),
            tf.TensorShape(([BATCH_SIZE]))
        )
    )

def preprocess(file,HEIGHT_WIDTH):
    imdata = mpimg.imread(file)

    imdata = cv2.resize(imdata, dsize=(HEIGHT_WIDTH, HEIGHT_WIDTH), interpolation=cv2.INTER_LINEAR) * 255.0
    imdata = tf.keras.applications.inception_resnet_v2.preprocess_input(
        imdata, data_format=None
    )

    return imdata, class_map[os.path.basename(os.path.dirname(file))]

def proko_train(model_class, epochs, num_ims_per_class,include_top=True,weights=None):
    print(f'starting script (num_ims_per_class={num_ims_per_class})')
    net = model_class(
        include_top=include_top,
        weights=weights,  # 'imagenet' <- Proko used imagenet,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1,  # 1000,2
        classifier_activation='sigmoid',
    )
    net.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    HEIGHT_WIDTH = net.input.shape[0]

    # overfitting?
    # look at both accuracy and val accuracy!
    train_data,test_data = get_data(num_ims_per_class)



    print(f'starting training (num ims per class = {num_ims_per_class})')
    history = net.fit(
        get_ds(train_data,HEIGHT_WIDTH),
        epochs=epochs,
        verbose=Verbose.PROGRESS_BAR,
        use_multiprocessing=False,
        shuffle=False,
        validation_data=get_ds(train_data,HEIGHT_WIDTH)
    )
    print('starting testing')
    print_output = True
    print(net.evaluate(
        get_ds(train_data,HEIGHT_WIDTH),
        verbose=Verbose.PROGRESS_BAR,
        use_multiprocessing=False
    ))
    print('script complete')
    return history

