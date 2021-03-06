from copy import deepcopy
from os.path import expanduser

import os

from mlib.boot.lang import ismac, listkeys
from mlib.job import Job
from mlib.JsonSerializable import obj
from mlib.str import lengthen_str
from mlib.term import log_invokation


@log_invokation
def make_jobs(cfg, muscle, experiments):
    return [build_job(
        experiment=None,
        cfg=cfg,
        muscle=muscle,
        gpus=None
    )] + [build_job(
        experiment=exp,
        cfg=cfg,
        muscle=muscle,
        gpus=exp.gpus
    ) for exp in experiments]

def build_job(
        experiment,
        cfg,
        muscle,
        gpus
):
    return DNN_Job(
        job_args={
            'tic'              : cfg.tic,

            'expid'            : '0' if experiment is None else experiment.expid,

            'arch'             : 'JUST_CLEAR_FILES_AND_GEN_DATASET' if experiment is None else experiment.arch,
            'ntrain'           : 0 if experiment is None else experiment.ntrain,

            'proto_model'      : cfg.proto_model,
            'pipeline'         : '' if experiment is None else cfg.PIPELINE.replace(' ', ''),

            'epochs'           : cfg.EPOCHS,
            'batchsize'        : 0 if experiment is None else cfg.BATCH_SIZE,
            'verbose'          : cfg.VERBOSE,
            'normtrainims'     : False if experiment is None else cfg.NORM_TRAIN_IMS,

            'salience'         : cfg.salience,
            'TRANSFER_LEARNING': cfg.TRANSFER_LEARNING,
            'REGEN_NTRAIN'     : cfg.REGEN_NTRAIN,
            'PRED_SIZE'        : cfg.PRED_SIZE,

            'deletenorms'      : cfg.OVERWRITE_NORMS if experiment is None else False,
            'gen'              : cfg.REGEN_DATA if experiment is None else False  # implies kill
        },
        job_cfg_arg=obj({
            'gen_cfg' : {
                'num_gpus'          : max(len(listkeys(muscle.GPU_IN_USE)), 2),
                'TRAINING_SET_SIZES': cfg.NTRAIN,
                'EVAL_SIZE'         : cfg.eval_nperc,
                'RSA_SIZE_PER_CLASS': cfg.rsa_nperc,
            } if experiment is None else None,
            'root'    : cfg.root,
            'full_cfg': deepcopy(cfg).toDict()  # REDUNDANT! also needs deepcopy because stupid toDict is in place
        }),
        gpus=gpus,  # [0,1,2,3] if RUN_EXPS_IN_SERIAL else, if empty is actually set to use all 4 in muscle
        interact=cfg.INTERACT,
        remote=not muscle.local,
        commands=[
            "rm -rf " + cfg.root,
            "find . -name \"*.pyc\" -exec rm -f {} \\;",
            "pkill -f miniconda3",
            "pkill -f MATLAB"
        ] if (experiment is None and not muscle.local) else [],
    )



class DNN_Job(Job):
    SUCCESS_STR = "NRC IS FINISHED"
    REMOTE_FOLDER = "~/mitili"
    REMOTE_SCRIPT = "src/main/python/exec/work.py"
    if ismac():
        LOCAL_PY = f'{expanduser("~")}/miniconda3/envs/dnn/bin/python'
    else:
        os.environ["CONDA_HOME"] = '/om2/user/mjgroth/miniconda3'
        LOCAL_PY = f'{os.environ["CONDA_HOME"]}/envs/dnn39/bin/python'
    LOCAL_SCRIPT = f'work.py'

    def __str__(self):
        if self.job_args['expid'] == '0':
            arch_str = ''
        else:
            arch_str = f'ARCH={self.job_args["arch"]}{self.job_args["ntrain"]}'
            arch_str = lengthen_str(arch_str, 20)
        return f'{super().__str__()}\tEXPID={self.job_args["expid"]}\t{arch_str}'
