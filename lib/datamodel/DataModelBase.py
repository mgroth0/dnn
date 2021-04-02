import dataclasses
from dataclasses import dataclass

import numpy as np
from typing import Dict, List, Optional, Union

class DataModelBase:
    pass



@dataclass
class RectangularMatrix:
    data: np.ndarray
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        return self.data[item]
    def __truediv__(self, other):
        return dataclasses.replace(self, data=(self.data / other))

@dataclass
class Named1DArrays(DataModelBase):
    data: Dict[str, Union[np.ndarray, List]]
    xlabel: str
    ylabel: str
    title_suffix: Optional[str] = None




class CategoricalUnorderedArrays(Named1DArrays): pass
class CategoricalOrderedArrays(Named1DArrays): pass
class CategoricalAlignedArrays(CategoricalOrderedArrays):
    def __post_init__(self):
        assert len(set([len(v) for v in self.data.values()])) == 1
