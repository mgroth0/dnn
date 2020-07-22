from dataclasses import dataclass

from lib.misc.imutil import resampleim
from mlib.boot.stream import arr
import numpy as np

from mlib.file import is_file


def preprocessors(hw): return {
    'none'           : Preprocessor(
        resize=hw
    ),
    'divstd_demean'  : Preprocessor(
        resize=hw,
        divstd=True,
        demean=True
    ),
    'unit_scale'     : Preprocessor(
        resize=hw,
        unit_scaling=True
    ),
    'demean_imagenet': Preprocessor(
        resize=hw,
        subtract_imagenet_means=True
    )
}


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
        if is_file(im):
            im = im.load()

        assert self.data_format == 'channels_last'
        assert self.channel_axis == 3
        assert self.nchan == 3

        if len(im.shape) == 3:
            return self._preprocess_im(im)
        elif len(im.shape) == 4:
            return arr([self._preprocess_im(i) for i in im])


    def _preprocess_im(self, img):
        import tensorflow as tf  # keep modular

        if self.resize is not None:
            img = resampleim(
                img, self.resize,
                self.resize,
                nchan=self.nchan
            )
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
