from mlib.boot.mlog import err
from mlib.input import boolinput, strinput
from mlib.web.api import API
from mlib.web.database import Database
def human_exp(cfg):

    from human_exps.mc_wait_pilot.mc_wait_pilot import MC_Wait_Pilot
    from human_exps.time_pilot.time_pilot import Time_Pilot
    from human_exps.contour_pilot.contour_pilot import Contour_Pilot
    exp = {
        'time_pilot'   : Time_Pilot,
        'mc_wait_pilot': MC_Wait_Pilot,
        'contour_pilot': Contour_Pilot,
    }[cfg.FLAGS[0]](_DEV=boolinput('dev'))
    command = strinput(f'what to do with {cfg.FLAGS[0]}', ['build', 'analyze'])
    if command == 'build':
        if boolinput('offline mode'):
            API.offline_mode = True
            Database.offline_mode = True
        if False:
            exp.DATABASE_IDS._hard_reset()
            exp.DATABASE_DATA._hard_reset()
        exp.build(
            _UPLOAD_RESOURCES=boolinput('upload resources'),
            _LOCAL_ONLY=boolinput('local only')
        )
    elif command == 'analyze':
        exp.analyze()
    else:
        err(f'unknown command: {command}')