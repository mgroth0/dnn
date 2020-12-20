from dataclasses import dataclass
from typing import Any
print('arch/__init__: about to import arches')
from arch.ALEXX import ALEX
from arch.GNET import GNET
from arch.INC import INC
from arch.SCRATCH import SCRATCH
from arch.assembled_model import AssembledModel
__all__ = ['ALEX', 'GNET', 'INC', 'SCRATCH', 'AssembledModel']
print('arch/__init__: finished')