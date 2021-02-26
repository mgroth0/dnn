import rsa_for_darius
print('top of dnn.py')
from lib import makereport
print('dnn.py: about to import log')
from mlib.boot import log
print('dnn.py: quarter through imports')
from mlib.boot.lang import isblank, ismac
from mlib.input import boolinput, strinput
print('dnn.py: half through imports')
from mlib.web import shadow
from mlib.file import Folder
from mlib.proj.struct import Project
from mlib.boot.stream import arr, listmap, __

log('defining DNN Project')
class DNN(Project):
    MODES = [
        'JUSTRUN',
        'PUSH',
        'COMPILE_TEST_ALL',
        'MAKEREPORT'
    ]
    HUMAN_EXPS_FOLDER = Folder('human_exps')
    if HUMAN_EXPS_FOLDER.exists:
        _human_exp_flags = listmap(__.name, HUMAN_EXPS_FOLDER.folders)
    else:
        _human_exp_flags = []
    extra_flags = _human_exp_flags + MODES
    def run(self, cfg):



        # import count_imagenet_data
        # count_imagenet_data.count()
        # return None



        # print('here1, doing Darius-RSA')
        # import rsa_for_darius
        # rsa_for_darius.main()
        rsa_for_darius.debug_process_post('AC')
        rsa_for_darius.test_line('AC')
        rsa_for_darius.debug_process_post('S')
        rsa_for_darius.test_line('S')
        rsa_for_darius.debug_process_post('NS')
        rsa_for_darius.test_line('NS')
        # print('here2, finished Darius-RSA')


        return None












        # keep modular
        assert not (cfg.REGEN_DATA and cfg.OVERWRITE_NORMS)  # btw, both imply killing worker before exp

        from mlib.boot.mlog import err
        from lib.dnn_lib import dnn
        from mlib.web.api import API
        from mlib.web.database import Database


        shadow.SHOW_INDEX = False

        if len(cfg.FLAGS) == 1 and cfg.FLAGS[0] in self._human_exp_flags:
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
        else:
            flag_mode = ''.join(arr(cfg.FLAGS).filtered(
                lambda s: s in self.MODES
            ))
            if not isblank(flag_mode): cfg.MODE = flag_mode
            if cfg.offline:
                API.offline_mode = True
                Database.offline_mode = True
                makereport.MAKEREPORT_ONLINE = False
            from mlib.km import kmscript  # keep modular
            if ismac():
                kmscript('activate run tool window')
            if isblank(cfg.MODE): cfg.MODE = ''.join(self.MODES)  # unnecessary?
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
