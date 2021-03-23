import numpy as np

from mlib.term import log_invokation


def make255(x):
    return np.uint8(x * 255)

def make1(x):
    return x / 255.0

make255 = np.vectorize(make255)
make1 = np.vectorize(make1)

import cv2
import numpy as np


def crop_or_pad(
        input_img: np.ndarray,
        newheight: int,
        axis=0
) -> np.ndarray:
    assert len(input_img.shape) == 3
    if input_img.shape[axis] > newheight:
        overby = input_img.shape[axis] - newheight
        if overby % 2 > 0:
            topcrop = ((overby - 1) / 2) + 1
        else:
            topcrop = overby / 2
        bottomcrop = int(overby / 2)
        _, input_img, _ = np.split(input_img, [int(topcrop), -bottomcrop], axis=axis)
        # input_img = input_img[int(topcrop):-bottomcrop, :]
    elif input_img.shape[axis] < newheight:
        underby = newheight - input_img.shape[axis]
        if underby % 2 > 0:
            toppad = ((underby - 1) / 2) + 1
        else:
            toppad = underby / 2
        bottompad = int(underby / 2)
        arg = [(0, 0), (0, 0), (0, 0)]
        arg[axis] = (int(toppad), bottompad)
        input_img = np.pad(input_img, arg, constant_values=0)
    return input_img


@log_invokation
def resampleim(im, heigh, width, nchan=1):
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
    imA.shape = (heigh, width, nchan)
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
