from collections import UserList
from dataclasses import dataclass

from typing import List

from mlib.boot.lang import enum





@dataclass
class Class:
    name: str
    index: int
    def __eq__(self, other):
        return other.index == self.index
    def __ne__(self, other):
        return not self == other
#     the term 'label' is ambiguous, and it confuses me if it refers to the string name or integer index so I'll avoid it

@dataclass
class ClassSet(UserList):
    classes: List[Class]
    def __post_init__(self):
        for i, c in enum(self.classes):
            assert c.index == i
        assert len(set([c.name for c in self.classes])) == len(self.classes)
        self.data = self.classes
