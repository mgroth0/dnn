

from lib.nn_main import nnet_main
print('top of dnn.py')

print('dnn.py: about to import log')
from mlib.boot import log
print('dnn.py: quarter through imports')
from mlib.boot.lang import isblank, ismac
print('dnn.py: half through imports')
# breakpoint()
from mlib.file import Folder
from mlib.web import shadow
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
    extra_flags = _human_exp_flags + MODES + [
        'IMAGENET_COUNT',
        'RSA_MAIN',
        'RSA_NEW',
        'ASD'
    ]
    def run(self, cfg):
        # keep modular
        assert not (cfg.REGEN_DATA and cfg.OVERWRITE_NORMS)  # btw, both imply killing worker before exp

        from lib.dnn_lib import dnn
        from mlib.web.api import API
        from mlib.web.database import Database


        shadow.SHOW_INDEX = False

        if len(cfg.FLAGS) == 1 and cfg.FLAGS[0] == 'IMAGENET_COUNT':
            import count_imagenet_data
            count_imagenet_data.count()
            return None
        elif len(cfg.FLAGS) == 1 and cfg.FLAGS[0] == 'RSA_MAIN':
            # print('here1, doing Darius-RSA')
            # import rsa_for_darius
            import rsa_for_darius
            rsa_for_darius.main()
            print('printing dnn is finished string!')
            print('__DNN_IS_FINISHED__')
            return None
        elif len(cfg.FLAGS) == 1 and cfg.FLAGS[0] in self._human_exp_flags:
            from lib.human_exp import human_exp
            human_exp(cfg)
        else:
            flag_mode = ''.join(arr(cfg.FLAGS).filtered(
                lambda s: s in self.MODES
            ))
            if not isblank(flag_mode): cfg.MODE = flag_mode
            if isblank(cfg.MODE): cfg.MODE = ''.join(self.MODES)  # unnecessary?
            if cfg.offline:
                API.offline_mode = True
                Database.offline_mode = True
                from lib import makereport
                makereport.MAKEREPORT_ONLINE = False
            from mlib.km import kmscript  # keep modular
            if ismac():
                kmscript('activate run tool window')
            dnn(cfg)
            print('__DNN_IS_FINISHED__')



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
