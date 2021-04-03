import os

from mlib.fig.PlotData import FigData
from mlib.file import Folder
os.chdir('..')
from lib import rsa_comp
from lib.rsa_comp_helpers import rsa_pattern, SYM_CLASS_SET_PATTERNS
from lib.rsa_figs import RSAImagePlot

# go  into a new, empty directory
# clone  https://github.com/mgroth0/dnn.git
# clone https://github.com/mgroth0/mlib
# cd dnn
# PYTHONPATH=.:../mlib python3 main/rsa.py

# set this to the path where you want figures to go
FigData.FOLDER = Folder('figs')

if __name__ == '__main__':
    for pat in SYM_CLASS_SET_PATTERNS.keys():
        RSAImagePlot(
            tags=['PATTERN', pat],
            comp_mat=rsa_pattern(pat, 10),
            net=f'PATTERN-{pat}',
            layer='PATTERN',
            pattern=pat,
            arch='PATTERN'
        ).save().build()
    def rsa_main(self, _CFG):
        rsa_comp.main(
            N_PER_CLASS=5,
            ACT_SIZE=10,
            INCLUDE_DARIUS=True,
            ALL_METHODS=False,
            EXCLUDE_DARIUS_SMALLER_TRAIN_SIZES=True,
            MULTIPROCESS=False,
            GPU=False
        )
