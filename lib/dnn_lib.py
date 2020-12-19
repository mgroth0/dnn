from lib.analyze_exp_group import analyze_exp_group
from lib.dnn_jobs import make_jobs
from lib.dnn_proj_struct import get_last_exp_group, get_figs_folder, DNN_ExperimentGroup, experiments_from_cfg
from lib.muscle import Muscle
from mlib import answer_request
from mlib.boot import log, mlog
from mlib.boot.lang import pwd
from mlib.file import File, SyncedFolder, Folder
from mlib.proj.struct import Project
from mlib.term import log_invokation


@log_invokation
def dnn(
        cfg
):
    mode = cfg.MODE
    log(f'MODE IS {mode}')
    Project.DNN_FIGS_FIGS_FOLDER.mkdirs()

    TEMP_FIGS_FOLDER = Folder(cfg.root)
    last_eg = get_last_exp_group()
    new_eg = None
    new_fig_folder = None

    muscle = Muscle(
        local=cfg.MUSCLE == 'local'
    )

    if cfg.CLEAR_EG_DATA:
        Project.DNN_FIGS_FIGS_FOLDER.clear()

    if 'JUSTRUN' in mode and cfg.SAVE_DATA:
        TEMP_FIGS_FOLDER.mkdir().clear()
        if 'next_exp_id' not in Project.STATE:
            Project.STATE['next_exp_id'] = 1
        if 'last_submitted_exp_group_name' not in Project.STATE:
            Project.STATE['last_submitted_exp_group_name'] = ''
        def check(a):
            Project.STATE["last_submitted_exp_group_name"] = a
            figs_folder, message = get_figs_folder(a)
            return figs_folder is not None, figs_folder if figs_folder is not None else message
        if cfg.EXPERIMENT_NAME is None:
            new_fig_folder = answer_request.answer_request(
                Project.STATE["last_submitted_exp_group_name"],
                "Experiment Name:",
                check,
                gui=cfg.GUI
            )
        else:
            new_fig_folder = check(cfg.EXPERIMENT_NAME)[1]
        new_fig_folder = File(new_fig_folder)
        log(f'{new_fig_folder=}')

    if 'JUSTRUN' in mode or 'PUSH' in mode:
        if cfg.MUSCLE != 'local':
            SyncedFolder(pwd()).sync(config='mitili', lpath='mitili')
        if 'JUSTRUN' in mode:
            cfg.tic = str(mlog.TIC)
            experiments = experiments_from_cfg(
                cfg,
                advance_id=True
            )
            jobs = make_jobs(cfg, muscle=muscle, experiments=experiments)
            muscle.run_all_jobs_main(
                jobs,
                serial=cfg.RUN_EXPS_IN_SERIAL,
                gui=cfg.GUI
            )
            temp_eg = DNN_ExperimentGroup.temp(TEMP_FIGS_FOLDER)
            temp_eg.save_md(cfg)

        if cfg.SAVE_DATA:
            new_eg = muscle.pull_data(TEMP_FIGS_FOLDER, cfg, new_fig_folder)

    exp_group = new_eg or last_eg
    log(f'MODE IS {mode}')
    if 'COMPILE_TEST_ALL' in mode:
        log('in CTA!')
        analyze_exp_group(exp_group)

        # the stuff below is only temporarily commented out
        # makefigs(exp_group.compile_folder, cfg.fig_backend, overwrite=True)
    # if 'MAKEREPORT' in mode:
    #     makereport.makereport()
