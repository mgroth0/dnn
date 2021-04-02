import dataclasses
from copy import deepcopy
from dataclasses import dataclass

from typing import List, Optional

from lib.datamodel.Mats import ComparisonMatrix
from lib.datamodel.Statistical import StatisticalArrays
from lib.fig_builder import FigBuilder
from lib.RSA_sym_model import RSA_CLASSES, RSA_LAYERS
from mlib.boot.dicts import dict_to_table, scrunch
from mlib.boot.lang import listkeys
from mlib.boot.stream import itr, listitems
from mlib.fig.TableData import TableData
from mlib.math import sigfig

@dataclass
class RSACorrCoefTable(FigBuilder):
    net_coefs: dict
    method_name: str
    EXCLUDE_DARIUS_SMALLER_TRAIN_SIZES: bool
    def key(self) -> List[str]:
        return [
            'pattern',
            'correlation',
            'coefficients',
            self.method_name
        ]
    def build(self):
        top = [''] + listkeys(self.net_coefs)
        orig_top = deepcopy(top)
        if self.EXCLUDE_DARIUS_SMALLER_TRAIN_SIZES:
            top[1:] = [s.split('_')[0] for s in top[1:]]
        full = [top]
        first = self.net_coefs[orig_top[1]]
        left = [pattern_strings[k] for k in first.keys()]
        cols = [left]
        for coefs in self.net_coefs.values():
            col = [sigfig(v, 3) for v in coefs.values()]
            cols.append(col)
        for i in itr(left):
            row = [col[i] for col in cols]
            full.append(row)
        TableData(
            data=full,
            title=f"Correlation Coefficients Between {method_strings[self.method_name]} of model activations and Perfect Classifier Patterns",
            title_size=70
        ).draw(builder=self, tags=self.tags + ['table', 'CorrCoefTable'])

@dataclass
class RSAPValueTable(FigBuilder):
    pat: str
    arch: str
    ttest_result: dict
    method_name: str
    net: str
    def key(self) -> List[str]:
        return [
            self.net,
            'pvalues',
            self.method_name,
            self.pat
        ]
    def build(self):
        only_two_tailed = scrunch(self.ttest_result, '-')
        for k, v in listitems(only_two_tailed):
            del only_two_tailed[k]['less']
            del only_two_tailed[k]['greater']
        TableData(
            data=dict_to_table(only_two_tailed)[1:],  # remove ['','two-sided']
            title=f'{self.net}: T-Test P-values Between Groupings of {method_strings[self.method_name]} Results, Grouped by {pattern_strings[self.pat]}',
            fontsize=40.0
        ).draw(builder=self, tags=self.tags + ['table', 'PValueTable', self.pat])

@dataclass
class RSAViolinBuilder(FigBuilder):
    sas: StatisticalArrays
    pat: str
    sim_string: str
    method_name: str
    arch: str
    net: str
    def key(self) -> List[str]:
        return [
            'violin',
            self.method_name,
            self.pat,
            self.net
        ]
    def build(self):
        fd = self.sas.violin()
        fd.title = f'{self.arch} {self.sim_string} Scores ({method_strings[self.method_name]}) Within and Between Classes Grouped by {pattern_strings[self.pat]}'
        fd.draw(builder=self, tags=self.tags + ['Violin', self.pat])

@dataclass
class RSAImagePlot(FigBuilder):
    comp_mat: ComparisonMatrix
    layer: str
    net: str
    arch: str
    block_reduce: bool = False
    log_scale: bool = False
    pattern: Optional[str] = None
    def key(self) -> List[str]:
        return [
            self.comp_mat.method_used,
            # 'of',
            # self.layer,
            'from',
            self.net,
            "_".join(self.cfgs()),
            self.comp_mat.method_used
        ]
    def cfgs(self):
        c = []
        if self.block_reduce:
            c.append('avg')
        if self.log_scale:
            c.append('log')
        return c
    def build(self):
        import numpy as np
        comp_mat_for_fig = self.comp_mat
        if self.block_reduce:
            import skimage.measure
            comp_mat_for_fig = dataclasses.replace(self.comp_mat, data=np.repeat(np.repeat(skimage.measure.block_reduce(
                self.comp_mat.data,
                tuple([int(len(self.comp_mat.data) / len(RSA_CLASSES))] * 2),
                np.mean
            ), len(RSA_CLASSES), axis=0), len(RSA_CLASSES), axis=1))
        fd = comp_mat_for_fig.image_plot()
        if self.log_scale:
            fd.y_log_scale = True

        if self.pattern:
            fd.title = f"{pattern_strings[self.pattern]} Classifier"
        else:
            fd.title = f'{method_strings[self.comp_mat.method_used]} of {self.arch} Activations'

            # fd.title += f' of {self.layer} from {self.net} ({",".join(self.cfgs())}) ({self.comp_mat.method_used})'
        fd.draw(builder=self, tags=self.tags + ['ImagePlot'])

pattern_strings = {
    'width': 'Band-Width',
    'band' : 'Band-Presence',
    'dark' : 'Darkness',
    'sym'  : 'Symmetry'
}
method_strings = {
    'L2_Norm': 'L2 Norm'
}
