from dataclasses import dataclass

import numpy as np
from pandas import DataFrame
from sklearn.cluster import KMeans
from typing import List, Optional, Type

from lib.datamodel.Classification import Class, ClassSet
from lib.datamodel.Correlational import Correlation, PearsonCorrelation
from lib.datamodel.DataModelBase import RectangularMatrix
from lib.misc import imutil
from mlib.boot.lang import islinux
from mlib.boot.stream import arr, itr, listmap, sort_human, zeros
from mlib.fig.TableData import RSAMatrix
from mlib.mat import rel_mat
from mlib.math import nan_above_eye, naneye
@dataclass
class FeatureMatrix(RectangularMatrix):
    #     rows are observations, columns are features
    class_set: ClassSet
    ground_truth: List[Class]
    def __post_init__(self):
        if self.ground_truth is not None:
            assert len(self.ground_truth) == len(self.data)
    def sort_by_class_name(self):
        assert self.ground_truth is not None
        zipped = sort_human(zip(self.data.tolist(), self.ground_truth), keyparam=lambda p: p[1].name)
        templist = self.ground_truth = []
        for z in zipped:
            templist.append(z[0])
            self.ground_truth.append(z[1])
        self.data = arr(templist)

    def compare(self, fun: Type[Correlation], GPU=False):
        special_confuse_mat = zeros(len(self.data), len(self.data))

        if (fun == PearsonCorrelation) and any([min(x) == max(x) for x in self.data]):
            raise MathFail
        #     # Pearson's Correlation Coefficient fails if
        #     # two arrays are commpared that have a zero standard deviation product (divide by zero)
        #     # Using an if statement above, I should prevent this

        data = self.data  # pleasework
        def _fun(i):  # cannot be lambda?
            return [(i, j, fun.fun(data[i, :], data[j, :])) for j in itr(data)]
        def _fun_tf(data):  # cannot be lambda?
            return fun.fun_tf(data)

        MULTIPROCESS = False

        from pathos.multiprocessing import ProcessPool


        if islinux() and MULTIPROCESS:
            #     slower than GPU
            #     BUGGY
            #     not optimized

            with ProcessPool() as p:
                # if islinux():
                # mapid = randrange(0,10000)
                # print(f'starting map {mapid}')
                r = p.map(_fun, itr(self.data))
            for results in r:
                for rr in results:
                    special_confuse_mat[rr[0], rr[1]] = rr[2]

        elif islinux() and GPU:
            import tensorflow as tf
            special_confuse_mat = tf.zeros((len(self.data), len(self.data)))







            with tf.device('/GPU:0'):
                special_confuse_mat = _fun_tf(self.data).numpy()

            # results[net] = rsa.numpy()
            # tfdata = tf.convert_to_tensor(self.data).cuda()

        else:
            r = listmap(_fun, itr(self.data))

            for results in r:
                for rr in results:
                    special_confuse_mat[rr[0], rr[1]] = rr[2]

        return ComparisonMatrix(
            data=nan_above_eye(naneye(special_confuse_mat)),
            method_used=fun.__name__,
            ground_truth=self.ground_truth,
            class_set=self.class_set
        )

    def kcompare(self, k=2):
        """Dissimilarity measured by whether each observation is in the same cluster or not"""
        return ComparisonMatrix(
            data=nan_above_eye(naneye(rel_mat(
                lambda x, y: x != y,
                KMeans(n_clusters=k).fit(DataFrame(self.data)).labels_
            ))),
            method_used="k-means clustering",
            ground_truth=self.ground_truth,
            class_set=self.class_set
        )


@dataclass
class ComparisonMatrix(RectangularMatrix):
    method_used: str
    ground_truth: Optional[List[Class]] = None
    class_set: Optional[ClassSet] = None
    def __getitem__(self, item):
        if isinstance(item, tuple) and isinstance(item[0], Class):
            return self.data[
                np.where(arr(self.ground_truth) == item[0], True, False),
                np.where(arr(self.ground_truth) == item[1], True, False)
            ]
        else:
            return super().__getitem__(item)
    def resize_if_longer_than(self, min):
        if len(self.data) > min:
            return self.resize_in_place(min)
        else:
            return self
    def resize_in_place(self, forced_res):
        if forced_res == len(self.data): return self
        self.ground_truth = self.ground_truth[::(int(self.data.shape[0] / forced_res))]
        self.data = imutil.resampleim(
            self.data,
            forced_res,
            forced_res,
            nchan=1
        )[:, :, 0]
        return self
    def image_plot(self):
        return RSAMatrix(
            data=self.data.tolist(),
            confuse_max=np.nanmax(self.data),
            confuse_min=np.nanmin(self.data),
            classnames=[c.name for c in self.class_set],
            title=f'{self.method_used}'
        )


class MathFail(Exception):
    pass
