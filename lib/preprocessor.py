from dataclasses import dataclass

import numpy as np


from lib.misc.imutil import make255, resampleim
from mlib.boot import log
from mlib.boot.mlog import err
from mlib.boot.stream import arr
from mlib.file import is_file


SAVE_PREPROCESSED = True


def preprocessors(hw): return {
    'none'                : Preprocessor(
        resize=hw
    ),
    'divstd_demean'       : Preprocessor(
        resize=hw,
        divstd=True,
        demean=True
    ),
    'unit_scale'          : Preprocessor(
        resize=hw,
        unit_scaling=True
    ),
    'demean_imagenet'     : Preprocessor(
        resize=hw,
        subtract_imagenet_means=True
    ),
    'demean_imagenet_crop': Preprocessor(  # recommended in convnets_keras for alexnet
        resize=hw + 29,  # so it becomes recommended 256 for alexnet but also works for other networks
        crop=hw,
        subtract_imagenet_means=True
    ),
    'INC_DEBUG'           : Inc_debug_preprocess()
}

import cv2

class Inc_debug_preprocess():
    def preprocess(self, im):
        file = None
        if is_file(im):
            file = im
            im = im.load()
        # imdata = mpimg.imread(file)
        from arch.INC_ORIG import INC_HW
        imdata = cv2.resize(im, dsize=(INC_HW, INC_HW), interpolation=cv2.INTER_LINEAR) * 255.0
        import tensorflow as tf
        imdata = tf.keras.applications.inception_resnet_v2.preprocess_input(
            # imdata = tf.keras.applications.inception_resnet_v2.preprocess_input(
            imdata, data_format=None
        )
        return imdata


@dataclass
class Preprocessor:
    resize: int = None
    crop: int = None
    divstd: bool = False
    subtract_imagenet_means: bool = False
    demean: bool = False
    unit_scaling: bool = False
    nchan: int = 3
    channel_axis: int = 3
    data_format: str = 'channels_last'

    def preprocess(self, im):
        log('starting preprocess')
        file = None
        if is_file(im):
            file = im
            im = im.load()

        assert self.data_format == 'channels_last'
        assert self.channel_axis == 3
        assert self.nchan == 3

        log('starting preprocess ops')
        if len(im.shape) == 2:
            im = np.stack((im, im, im), axis=2)
            return self._preprocess_im(im, file)
        elif len(im.shape) == 3:
            return self._preprocess_im(im, file)
        elif len(im.shape) == 4:
            err('maybe this is the problem?')
            return arr([self._preprocess_im(i, file) for i in im])
        else:
            err('or this?')



    def _preprocess_im(self, img, file):
        import tensorflow as tf  # keep modular

        needs_resize = (img.shape[0] != self.resize) or (img.shape[1] != self.resize)

        # breakpoint()
        if needs_resize and self.resize is not None:
            img = resampleim(
                img,
                self.resize,
                self.resize,
                nchan=self.nchan
            )
            if file is not None:
                # breakpoint()
                file.save(make255(img))
        if self.crop:
            if self.channel_axis == 1:
                img = img[:, (self.resize - self.crop) // 2:(self.resize + self.crop) // 2
                , (self.resize - self.crop) // 2:(self.resize + self.crop) // 2]
            else:
                img = img[(self.resize - self.crop) // 2:(self.resize + self.crop) // 2
                , (self.resize - self.crop) // 2:(self.resize + self.crop) // 2, :]
        if not issubclass(img.dtype.type, np.floating):
            img = img.astype(tf.keras.backend.floatx(), copy=False)
        if self.unit_scaling:
            img /= 127.5
            img -= 1.
        if self.divstd:
            img[:, :, 0] /= np.std(img[:, :, 0])
            img[:, :, 1] /= np.std(img[:, :, 1])
            img[:, :, 2] /= np.std(img[:, :, 2])
        if self.subtract_imagenet_means:
            img[:, :, 0] -= 123.68
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 103.939
        if self.demean:
            img[:, :, 0] -= np.mean(img[:, :, 0])
            img[:, :, 1] -= np.mean(img[:, :, 1])
            img[:, :, 2] -= np.mean(img[:, :, 2])
        if self.data_format == 'channels_first':
            img = np.moveaxis(img, 2, 0)
        return img
