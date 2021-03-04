from abc import abstractmethod

from mlib.analyses import Analysis

class DNN_Analysis(Analysis):
    def after_build(self, FLAGS, net): pass
    def after_val(self, i, net, nam): pass
    def after_fit(self, i, net, nam): pass
    @abstractmethod
    def during_compile(self, eg): pass
    @abstractmethod
    def get_report_figdata(self, exp_name, resources_root, database): pass

class PostBuildAnalysis(DNN_Analysis):
    @abstractmethod
    def after_build(self, FLAGS, net): pass



class PerEpochAnalysis(DNN_Analysis):
    @abstractmethod
    def after_val(self, i, net, nam): pass
    @abstractmethod
    def after_fit(self, i, net, nam): pass
