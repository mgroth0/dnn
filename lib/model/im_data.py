from mlib.boot.lang import ismac
from mlib.boot.mlog import err
from mlib.file import Folder

OM_IM_DATA_ROOT = Folder('/om2/user/mjgroth/data/im')
_RAW = 'raw'
class ImageDataset:
    def __init__(self, name):
        self.folder = OM_IM_DATA_ROOT[name]
        self.metadata_file = self.folder['metadata.json']
        self.raw = ImageDatasetTransformation(self, _RAW)
        self.remote = ismac()
        if not self.remote:
            self.verify()
    def metadata(self):
        return self.metadata_file.load()
    def verify(self):
        assert self.folder.exists
        assert self.metadata_file.exists



class ImageDatasetTransformation:
    def __init__(self, dataset: ImageDataset, name: str):
        self.dataset = dataset
        self.remote = dataset.remote
        if name == _RAW:
            self.folder = self.dataset.folder[_RAW]
        else:
            self.folder = self.dataset.folder['trans'][name]
        self.train = ImageDatasetSplit(self, 'train')
        self.test = ImageDatasetSplit(self, 'test')
        self.eval = ImageDatasetSplit(self, 'eval')
        if not self.remote:
            self.verify()
    def verify(self):
        assert self.folder.exists

    def copy_sample_images(self, destination: Folder, n_per_class=10):
        return self.train.copy_sample_images(destination, n_per_class)

class ImageDatasetSplit:
    def __init__(self, trans: ImageDatasetTransformation, name):
        self.trans = trans
        self.remote = trans.remote
        self.folder = trans.folder[name]

        if not self.remote:
            self.verify()
    def verify(self):
        assert self.folder.exists
    def copy_sample_images(self, destination: Folder, n_per_class):
        import random
        random
        for class_folder in self.folder.files:
            if ismac(): err('cant do this locally!')




Symmetry = ImageDataset('sym')
Carnivora = ImageDataset('carnivora')
