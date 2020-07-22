import math

import tensorflow.keras
import tensorflow as tf

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.
from mlib.boot.stream import arr

class MySequence(tensorflow.keras.utils.Sequence):

    def __init__(self, ims, batch_size):
        self.len = len(ims)
        # data = catfun(lambda x: x,ims.arrayof().data).shape
        self.ims = tf.data.Dataset.from_tensor_slices(ims.arrayof().data)
        self.ys = ims.arrayof().label
        # self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.len / self.batch_size)

    def __getitem__(self, idx):
        idxs = range(idx * self.batch_size, (idx + 1) * self.batch_size)
        datas = []
        labels = []
        idx = -1
        for x in self.ims.as_numpy_iterator():
            idx = idx + 1
            if idx in idxs:
                datas.append(x)
                labels.append(self.ys[idx])
        datas = arr(datas)
        labels = arr(labels)

        # batch_y = self.y[idx * self.batch_size:(idx + 1) *
        #                                        self.batch_size]

        return datas, labels
        # return np.array([
        #     resize(imread(file_name), (200, 200))
        #     for file_name in batch_x]), np.array(batch_y)
