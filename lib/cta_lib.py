from abc import abstractmethod, ABC

from lib.defaults import *

class CompiledResult(ABC):
    def __init__(
            self,
            dims,
            suffix,
            is_table=False,
            rows=None,
            cols=None
    ):
        self.dims = dims
        self.row_headers = None
        self.col_headers = None
        if dims == 1:
            self.data = arr2d()
        elif dims == 2:
            if isinstance(self, FinalResult):
                self.data = np.zeros((len(rows), len(cols), 1))
                self.row_headers = make2d([None] + rows).T
                self.col_headers = make2d([None] + cols)
            else:
                self.data = arr3d()
        self.suffix = suffix
        self.j = None
        self.is_table = is_table
    @abstractmethod
    def exp_data(self, exp): pass
    @abstractmethod
    def append(self, data, indices, is_GNET=False, ): pass
    def _file(self, prefix):
        f = File(prefix + self.suffix)
        if self.j is None:
            self.j = f.load(as_object=True)
            if self.dims == 1:
                self.template = self.j.viss[0]
                self.j.viss = self.j.viss[1:]
        return f



class AverageResult(CompiledResult):
    def exp_data(self, exp):
        return self._file(exp.prefix).viss[0]
    def append(self, exp_data, indices, is_GNET=False):
        if self.dims == 1:
            row = exp_data.y
            if is_GNET:
                row = np.mean(np.reshape(arr(row), (-1, 3)), axis=1).tolist()
            self.data = vert(self.data, row)
        elif self.dims == 2:
            data = arr(exp_data.data)
            if self.is_table:
                data = arr(exp_data.data, dtype=np.object)
                self.row_headers = make2d(data[:, 0]).T
                self.col_headers = make2d(data[0, :])
                data = arr(data[1:, 1:], dtype=float)
            self.data = lay(self.data, data)
class FinalResult(CompiledResult):
    def append(self, data, indices, is_GNET=False, ):
        self.data[indices] = data

    def exp_data(self, exp):
        return self._file(exp.prefix).viss[0].y[-1]

# class ExampleResult(CompiledResult):
#     def __init__(self, suffix):
#         super(ExampleResult, self).__init__(
#             dims=None,
#             suffix=suffix
#         )
#     def append(self, data, indices, is_GNET=False, ):
#         self.data[indices] = data
#
#     def exp_data(self, exp):
#         return self._file(exp.prefix).viss[0].y[-1]
