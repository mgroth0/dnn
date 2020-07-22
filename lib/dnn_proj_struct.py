from dataclasses import dataclass

from typing import Optional

from lib.nn_main import ARCH_MAP
from mlib.boot.mlog import err, warn
from mlib.boot.stream import arr
from mlib.file import Folder, MD_FILE
from mlib.proj.struct import Project

def experiments_from_folder(f):
    r = []
    for exp_folder in Folder(f).files:
        if exp_folder.name == MD_FILE: continue
        exp_folder = Folder(exp_folder)
        r += [DNN_Experiment(
            expid=exp_folder.name,
            arch=exp_folder.md_file['arch'],
            ntrain=exp_folder.md_file['ntrainim'],
            gpus=None,
            folder=Folder(exp_folder)
        )]
    return arr(r)

def experiments_from_group(exp_group): return experiments_from_folder(exp_group.folder)


def experiments_from_cfg(cfg, advance_id):
    experiments = []
    for i in range(cfg.REPEAT_ALL):
        for exp in cfg.EXPS:
            for ntrain in cfg.NTRAIN:
                if advance_id:
                    exp_id = str(Project.STATE["next_exp_id"])
                    Project.STATE["next_exp_id"] = int(exp_id) + 1
                else:
                    exp_id = None
                    err('not ready to handle this')
                experiments += [DNN_Experiment(
                    arch=exp.arch,
                    ntrain=ntrain,
                    expid=exp_id,
                    gpus=exp.gpus,
                    folder=None
                )]
    return arr(experiments)


def get_last_exp_group():
    if experiment_groups().isnotempty:
        return experiment_groups().max_by(lambda e: e.index)
    else:
        return None

def get_figs_folder(nameRequest):
    egs = experiment_groups()
    if '-' in nameRequest: return None, 'cannot use "-" in name'
    if nameRequest in [e.name for e in egs]: return None, f'{nameRequest} was already used!'
    if egs.isnotempty:
        num = max([e.index for e in egs]) + 1
    else:
        num = 0
    return Project.DNN_FIGS_FIGS_FOLDER[f'{num}-{nameRequest}'], None

def experiment_groups():
    egs = []
    for f in Project.DNN_FIGS_FIGS_FOLDER.files:
        if len(f.name.split('-')) == 2:
            egs.append(
                DNN_ExperimentGroup.from_folder(f)
            )
    return arr(egs)



@dataclass
class DNN_ExperimentGroup:
    folder: Folder
    index: Optional[int]
    name: Optional[str]
    _is_temp: bool = False

    @staticmethod
    def temp(temp_folder):
        return DNN_ExperimentGroup(
            index=None,
            name=None,
            folder=temp_folder,
            _is_temp=True
        )
    def save_md(self, cfg):
        self.folder['metadata.json'].save({
            'archs'     : [{
                'label'      : e.arch,
                'description': '(pre-trained on ImageNet)' if ARCH_MAP[e.arch]().IS_PRETRAINED else '(untrained)'
            } for e in cfg.EXPS],
            'nrepeats'  : cfg.REPEAT_ALL,
            'nepochs'   : cfg.EPOCHS,
            'batchsize' : cfg.BATCH_SIZE,
            'ntrainims' : cfg.NTRAIN,
            'normalized': cfg.NORM_TRAIN_IMS
        })

    @staticmethod
    def from_folder(f):
        if len(f.name.split('-')) != 2:
            err(f'{f} is not named right: {f.name.split("-")}')
        return DNN_ExperimentGroup(
            index=f.name.split('-')[0],
            name=f.name.split('-')[1],
            folder=Folder(f)
        )

    def __post_init__(self):
        self.index = int(self.index) if self.index is not None else None
        self.compile_folder = self.folder.res_pre_ext('-compile')
        self.compile_exp_res_folder = self.compile_folder['1/compiled']
        self.priv_web_folder = Project.DNN_WEB_FOLDER[self.folder.name + '-web-priv']
        self.pub_web_folder = Project.DNN_WEB_FOLDER[self.folder.name + '-web-pub']

    @property
    def compiled(self):
        return self.compile_folder.exists

    @property
    def metadata(self):
        return self.folder['metadata.json']

    @property
    def experiments(self):
        return experiments_from_group(self)

@dataclass
class DNN_Experiment:
    arch: str
    ntrain: int
    expid: int
    gpus: Optional[list]
    folder: Optional[Folder]  # for completed experiments
