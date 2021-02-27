# 0=silent,1=progress bar,2=one line per epoch
from enum import Enum
class Verbose(Enum):
    SILENT = 0
    PROGRESS_BAR = 1
    PRINT_LINE_PER_EPOCH = 2
