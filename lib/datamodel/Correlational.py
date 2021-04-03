import numpy as np
from abc import ABC, abstractmethod
from typing import Dict

from lib.datamodel.DataModelBase import CategoricalAlignedArrays
from mlib.boot.mlog import err



class Correlation(ABC):

    @abstractmethod
    def HIGH_IS_SIMILAR(self): err('needs override')

    @classmethod
    def fun(cls, array1, array2):
        assert len(array1.shape) == 1 and len(array2.shape) == 1
        assert len(array1) == len(array2)
        return cls._fun(array1, array2)



    @staticmethod
    def _fun(array1, array2): pass

class L2_Norm(Correlation):
    HIGH_IS_SIMILAR = False
    def fun_tf(self,feature_mat):
        import tensorflow as tf
        data_tf = tf.math.l2_normalize(feature_mat, axis=1)
        return  tf.matmul(data_tf, tf.transpose(data_tf))
    @staticmethod
    def _fun(array1, array2):
        return np.linalg.norm(array1 - array2)

class PearsonCorrelation(Correlation):
    HIGH_IS_SIMILAR = True
    @staticmethod
    def _fun(array1, array2):
        # untested
        return np.corrcoef(array1, array2)[0][1]

class Covariance(Correlation):
    HIGH_IS_SIMILAR = True
    @staticmethod
    def _fun(array1, array2):
        # untested
        return np.cov(array1, array2)[0][1]

# class KMeansAsCorrelation(Correlation):
#     def setup(self):
#
#     @staticmethod
#     def _fun(array1, array2):
#         # untested


class CorrelationalArrays(CategoricalAlignedArrays):
    # pearsons correlation coefficient?
    def correlate_to(self, key) -> Dict[str, float]:
        import pandas as pd
        import numpy as np
        array1 = self.data[key]
        rest = []
        for k in self.data:
            if k != key:
                rest.append(k)
        # pandas is supposedly better at handling nans!! (I proved this true by trying the same math on masked numpy, which constantly threw errors
        coefs = {r: pd.concat(
            [pd.DataFrame(array1), pd.DataFrame(self.data[r])],
            axis=1).cov().iat[0, 1] / (np.nanstd(array1) * np.nanstd(self.data[r]))
                 for r in rest}
        return coefs
