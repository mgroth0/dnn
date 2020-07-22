from dataclasses import dataclass

from abc import ABC

from arch.model_wrapper import ModelWrapper
from mlib.abstract_attributes import Abstract

class PreAssembledModel(ModelWrapper, ABC):
    # @dataclass
    # class STATIC_ATTS(ModelWrapper.STATIC_ATTS):
    IS_PRETRAINED = Abstract(bool)
    ARCH_LABEL = Abstract(str)
    HEIGHT_WIDTH = Abstract(int)
