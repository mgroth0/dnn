from copy import deepcopy

import numpy as np

from lib.datamodel.DataModelBase import CategoricalUnorderedArrays
from mlib.boot.lang import listkeys, listvalues
from mlib.boot.stream import listitems
from mlib.fig.PlotData import PlotData

class StatisticalArrays(CategoricalUnorderedArrays):
    def bar(self):
        means = {k: np.nanmean(v) for k, v in self.data.items()}
        fd = PlotData(
            y=listvalues(means),
            item_type='bar',
            **self._common(),
        )
        return fd
    def violin(self):
        fd = PlotData(
            y=[y.tolist() for y in listvalues(self.data)],
            item_type='violin',
            **self._common(),
        )

        return fd

    def _common(self):
        return dict(
            x=listkeys(self.data),
            item_color=[[0, 0, b] for b in np.linspace(0, 1, len(self.data))],
            title=f'{self.ylabel} by {self.xlabel}' + (f' ({self.title_suffix})' if self.title_suffix else ''),
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            bar_sideways_labels=False,
            title_size=20.0
        )

    def ttests(self, redundant_full=False):
        from scipy import stats  # scipy doesnt auto import subpackages
        ALTS = ['two-sided', 'less', 'greater']
        i = 0
        r = {}
        for name, mat in listitems(self.data):
            c = 0
            for name2, mat2 in listitems(self.data):
                if c <= i:
                    c += 1
                    continue
                if name not in r: r[name] = {}
                pvalues = {alt: stats.ttest_ind(mat, mat2, alternative=alt)[1] for alt in ALTS}
                r[name][name2] = deepcopy(pvalues)
                if redundant_full:
                    if name2 not in r: r[name2] = {}
                    less = pvalues['less']
                    pvalues['less'] = pvalues['greater']
                    pvalues['greater'] = less
                    r[name2][name] = pvalues
                c += 1
            i += 1
        return r
