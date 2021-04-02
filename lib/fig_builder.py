from dataclasses import dataclass

from abc import abstractmethod
from typing import List

from mlib.fig.PlotData import FigData
@dataclass
class FigBuilder:
    tags: List[str]
    def save(self):
        FigData.FOLDER[self.safekey() + '.pkl'].save(self, silent=True)
        return self
    def safekey(self):
        LATEX_SAFE_CHAR = '-'
        # turns out '_' is not safe!
        return '-'.join(self.key()).replace(
            ' ', LATEX_SAFE_CHAR
        ).replace(
            '(', LATEX_SAFE_CHAR
        ).replace(
            ')', LATEX_SAFE_CHAR
        ).replace(
            ',', LATEX_SAFE_CHAR
        ).replace(
            '_', LATEX_SAFE_CHAR
        ).replace(
            ':', LATEX_SAFE_CHAR
        ).replace(
            '[', LATEX_SAFE_CHAR
        ).replace(
            ']', LATEX_SAFE_CHAR
        ).replace(
            '/', LATEX_SAFE_CHAR
        ).replace(
            '\\', LATEX_SAFE_CHAR
        )
    @abstractmethod
    def key(self) -> List[str]: pass
    @abstractmethod
    def build(self): pass
