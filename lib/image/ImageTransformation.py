from dataclasses import dataclass

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

from mlib.file import File

@dataclass
class ImageTransformation(ABC):
    intermediate_hook: Optional[Callable[[str, np.ndarray], None]] = lambda name, img: None


    def transform(self, input: Union[np.ndarray, File]) -> np.ndarray:
        if isinstance(input, File):
            input = input.load()
        return self._transform(input)

    @abstractmethod
    def _transform(self, input: Union[np.ndarray, File]) -> np.ndarray:
        pass
