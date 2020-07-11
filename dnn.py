from human_exps.mc_wait_pilot.mc_wait_pilot import MC_Wait_Pilot

from human_exps.time_pilot.time_pilot import Time_Pilot
from mlib.boot.mlog import err
from mlib.boot.stream import arr, listmap, __
from mlib.file import Folder
from mlib.km import kmscript
from mlib.proj.struct import Project
from lib.dnn_lib import dnn
from mlib.web.api import API
from mlib.web.database import Database
from mlib.boot.mutil import boolinput, strinput
from sanity import sanity

class DNN(Project):
    MODES = [
        'CLEAN',
        'JUSTRUN',
        'PUSH',
        'GETANDMAKE',
        'MAKEFIGS',
        'COMPILE_TEST_ALL',
        'MAKEREPORT'
    ]
    HUMAN_EXPS_FOLDER = Folder('human_exps')
    _human_exp_flags = listmap(__.name, HUMAN_EXPS_FOLDER.folders)
    extra_flags = _human_exp_flags + MODES + ['sanity']
    def run(self, cfg):
        if len(cfg.FLAGS) == 1 and cfg.FLAGS[0] == 'sanity':
            sanity()
        elif len(cfg.FLAGS) == 1 and cfg.FLAGS[0] in self._human_exp_flags:
            exp = {
                'time_pilot'   : Time_Pilot,
                'mc_wait_pilot': MC_Wait_Pilot,
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
        else:
            cfg.MODE = ''.join(arr(cfg.FLAGS).filtered(
                lambda s: s in self.MODES
            ))
            self.init()
            kmscript('activate run tool window')
            dnn(cfg)

    instructions = '''Generate some images, train/test a model, run analyses, and generate plots. Tested on Mac, but not yet on linux/Windows.

- `./dnn -cfg=gen_images --INTERACT=0`
- `./dnn -cfg=test_one --INTERACT=0`

The second command will fail with a Mathematica-related error, but your results will be saved in `_figs`.

TODO: have to also consider running and developing other executables here: human_exp_1 and human_analyze

'''
    configuration = '''-MODE: (default = FULL) is a string that can contain any combination of the following (example: "CLEAN JUSTRUN")
- CLEAN
- JUSTRUN
- GETANDMAKE
- MAKEREPORT

Edit [cfg.yml]() to save configuration options. Feel free to push these.

If there is anything hardcoded that you'd like to be configurable, please submit an issue.'''
    credits = 'Darius, Xavier, Pawan\n\nheuritech, raghakot, joel'
