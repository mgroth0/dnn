import math
import pdb

from PIL import Image
import numpy as np
import scipy

import tensorflow.keras

from mlib.boot.mlog import log
from mlib.boot.mutil import arr,catfun
def make255(x):
    return np.uint8(x * 255)

def make1(x):
    return x/255.0

make255 = np.vectorize(make255)
make1 = np.vectorize(make1)

import cv2
import numpy as np



def resampleim(im,heigh,width):

    # data = np.array(list(Image.fromarray(im).resize((width,heigh)).getdata()))

    # img = cv2.imread('your_image.jpg')
    # interpolation=cv2.INTER_CUBIC
    # interpolation=cv2.INTER_NEAREST
    # log('resampleimA',silent=True)
    imA = cv2.resize(im, dsize=(width, heigh), interpolation=cv2.INTER_LINEAR)
    # log('resampleimB',silent=True)
    # imB = cv2.resize(im, dsize=(width, heigh), interpolation=cv2.INTER_NEAREST)

    # im = Image.fromarray(im)
    # log('resampleim1',silent=True)
    # resized = im.resize((width,heigh))
    # log('resampleim2',silent=True)
    # data = resized.getdata()
    # log('resampleim3',silent=True)

    # data= list(data)

    # log('resampleim3.2',silent=True)

    # data= np.array(data)

    # log('resampleim3.5',silent=True)
    imA.shape = (heigh,width,1)
    # log('resampleim3.6',silent=True)
    # np.reshape(data,(heigh,width,1))
    return imA
    # img = Image.open('somepic.jpg')
    # wpercent = (basewidth/float(img.size[0]))
    # hsize = int((float(img.size[1])*float(wpercent)))
    # img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    #
    #
    # return scipy.ndimage.zoom(im, zoom=[heigh,width,1],order=0)


# from skimage.io import imread
# from skimage.transform import resize
# import numpy as np
# import math
import tensorflow as tf


# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

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
        idxs= range(idx * self.batch_size,(idx + 1) *self.batch_size)
        datas=[]
        labels=[]
        idx = -1
        for x in self.ims.as_numpy_iterator():
            idx = idx + 1
            if idx in idxs:
                datas.append(x)
                labels.append(self.ys[idx])
        datas=arr(datas)
        labels=arr(labels)

        # batch_y = self.y[idx * self.batch_size:(idx + 1) *
        #                                        self.batch_size]

        return datas,labels
        # return np.array([
        #     resize(imread(file_name), (200, 200))
        #     for file_name in batch_x]), np.array(batch_y)