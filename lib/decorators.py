from functools import wraps

from mlib.fig.PlotData import FigData
from mlib.file import Folder

class FolderBuilder:
    def __init__(self):
        assert FigData.FOLDER is None
        self.fold = Folder('temp/afolder', quiet=True).mkdirs().clear(silent=True)
        FigData.FOLDER = self.fold
    def finish(self):
        FigData.FOLDER = None
        return self.fold
def build_folder(f, *args, **kwargs):
    fb = FolderBuilder()
    f(*args, **kwargs)
    return fb.finish()
def folder_builder(f):
    @wraps(f)
    def ff(*args, **kwargs):
        return build_folder(f, *args, **kwargs)
    return ff
