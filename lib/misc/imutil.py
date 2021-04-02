import numpy as np

from mlib.boot.mlog import LogLevel
from mlib.term import log_invokation


def make255(x):
    return np.uint8(x * 255)

def make1(x):
    return x / 255.0

make255 = np.vectorize(make255)
make1 = np.vectorize(make1)




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

def resizeim(input_img, height, width, fit_dim=1):
    assert fit_dim in [0, 1]
    if fit_dim == 1:
        input_img = fitim(input_img, None, width)
    else:
        input_img = fitim(input_img, height, None)
    if fit_dim == 0:
        input_img = crop_or_pad(input_img, height, axis=0)
    else:
        input_img = crop_or_pad(input_img, width, axis=1)
    return input_img

def fitim(input_img, height, width):
    assert ((height is None) or (width is None)) and not ((height is None) and (width is None))
    assert len(input_img.shape) == 3
    if height is None:
        ratio = width / input_img.shape[1]
        return resampleim(input_img, int(input_img.shape[0] * ratio), width, nchan=3)
    else:
        ratio = height / input_img.shape[0]
        return resampleim(input_img, height, int(input_img.shape[1] * ratio), nchan=3)

@log_invokation(level=LogLevel.INFO)
def resampleim(im, heigh, width, nchan=1):
    import cv2  # 3 SECOND IMPORT
    imA = cv2.resize(im, dsize=(width, heigh), interpolation=cv2.INTER_LINEAR)
    imA.shape = (heigh, width, nchan)
    return imA
