from dataclasses import dataclass

import numpy as np
from typing import Optional

from mlib.boot.lang import esorted, ismac, listvalues
from mlib.datamodel.DataModelBase import Class, ClassSet
from mlib.file import File, Folder

OM_IM_DATA_ROOT = Folder('/om2/user/mjgroth/data/im')
_RAW = 'raw'
_TRAIN = 'train'
_TEST = 'test'
_EVAL = 'eval'
class ImageDataset:
    def __init__(self, name):
        self.folder = OM_IM_DATA_ROOT[name]
        self.metadata_file = self.folder['metadata.json']
        self.trans = {_RAW: ImageDatasetTransformation(self, _RAW)}
        self.splits = {n: ImageDatasetSplit(self, n) for n in [_TRAIN, _TEST, _EVAL]}
        if not ismac():
            self.classes = ClassSet([ImageDatasetClass(
                name=File(class_folder).name,
                index=i,
                dataset=self
            ) for i, class_folder in esorted(listvalues(self.splits)[0].folder(listvalues(self.trans)[0]).paths)])
        if not ismac():
            self.verify()
    def metadata(self):
        return self.metadata_file.load()
    def verify(self):
        self.folder.assert_exists()
        for t in self.trans.values():
            t.verify()
            for s in self.splits.values():
                s.verify(t)
                [c.verify(s, t) for c in self.classes if s.islabeled]


    def sample(self, n_per_class=10, preload=False):
        import random
        tran = listvalues(self.trans)[0]
        splt = listvalues(self.splits)[0]
        the_sample = []
        for cls in listvalues(self.classes):
            cls: ImageDatasetClass
            folder = cls.folder(splt, tran)
            im_paths = folder.paths
            num = len(im_paths)
            assert num > n_per_class
            already_took = []
            n = 0
            while n < n_per_class:
                i = random.randrange(0, num)
                if i in already_took:
                    continue
                else:
                    the_im = ImageDatasetImage(File(im_paths[i]), self, tran, splt, cls)
                    if preload:
                        the_im.load()
                    the_sample.append(the_im)
                    n += 1
                    already_took += [i]
        return the_sample


@dataclass
class ImageDatasetTransformation:
    dataset: ImageDataset
    name: str
    @property
    def folder(self):
        if self.name == _RAW:
            return self.dataset.folder[_RAW]
        else:
            return self.dataset.folder['trans'][self.name]
    def verify(self):
        self.folder.assert_exists()

@dataclass
class ImageDatasetSplit:
    dataset: ImageDataset
    name: str
    islabeled: bool = True
    def __post_init__(self):
        if self.name == _TEST:
            self.islabeled = False
    def folder(self, trans: ImageDatasetTransformation):
        return trans.folder[self.name]
    def verify(self, trans: ImageDatasetTransformation):
        self.folder(trans).assert_exists()

@dataclass
class ImageDatasetClass(Class):
    dataset: ImageDataset
    def folder(self, split: ImageDatasetSplit, trans: ImageDatasetTransformation):
        return split.folder(trans)[self.name]
    def verify(self, split: ImageDatasetSplit, trans: ImageDatasetTransformation):
        self.folder(split, trans).assert_exists()


@dataclass
class ImageDatasetImage:
    file: File
    dataset: ImageDataset
    trans: ImageDatasetTransformation
    split: ImageDatasetSplit
    cls: ImageDatasetClass
    data: Optional[np.array] = None
    def load(self):
        self.data = self.file.load(silent=True)
    @property
    def name(self):
        return self.file.name


if False:
    Symmetry = ImageDataset('sym')
Carnivora = ImageDataset('carnivora')
