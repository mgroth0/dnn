from lib import getfigdata
import lib.boot
from lib.defaults import *
from lib.figs.JsonSerializable import obj
from lib.gui import answer_request
from lib.misc.google_compute import gcloud_config

@log_invokation
def dnn(
        cfg
):
    TEMP_FIGS_FODLER = cfg.root
    mode = cfg.MODE
    FULL = 'FULL' in mode
    if 'CLEAN' in mode or FULL:
        shell('pkill -f miniconda3')
        shell('pkill -f MATLAB')
    import lib.run_exps as run_exps
    if FULL and cfg.SAVE_DATA:
        metastate = File("_metastate.json")
        def check(a):
            figs_folder = get_figs_folder(a)
            if figs_folder is None:
                return (False, f"{folName} was already used!")
            else:
                log('figs_folder:' + figs_folder.abspath)
                return (True,)
        figsFolder = answer_request.answer_request(metastate["last_submitted_exp_group_name"], "Experiment Name:",
                                                   check,
                                                   gui=cfg.GUI)
        metastate["last_submitted_exp_group_name"] = figsFolder
        figsFolder = File(figsFolder).abspath

        if not figsFolder:
            log('did not get a fig folder')
            exit(1)
        log('figsFolder=$', figsFolder)
    if 'CLEAN' in mode or FULL:
        log('cleaning')
        File(TEMP_FIGS_FODLER).deleteIfExists()
        File('_logs/local/dnn').deleteIfExists()
        File('_logs/remote/dnn').deleteIfExists()
    if FULL or 'JUSTRUN' in mode or 'PUSH' in mode:
        log('about to run justrun')
        log('in justrun')
        d = int(time.time())
        # File(pwd()+'/bin/tic.txt').write(d)
        log('pushing')

        if cfg.MUSCLE != 'local':
            gcloud_config()

            # this used to just readlines instead of interact. might break if no terminal?
            SyncedFolder(pwd()).sync(config='mitili', lpath='mitili')



            # child = SSHProcess('bash '+pwd()+'/bin/my_rsync',
            #                    timeout=None,
            #                    logfile_read=sys.stdout.buffer,
            #                    )
            #
            # def finishSync():
            #     # # this has to be called or it will block
            #     if child.alive():
            #         child.readlines()
            #         child.wait()
            # atexit.register(finishSync)
            # child.login()
            # finishSync()
            # child.interact()







            # child = SSHProcess('bash 'pwd()+'/bin/my_rsync2')
            # child.login()
            #
            #
            # child.interact()

        if FULL or 'JUSTRUN' in mode:
            log('running in gc')
            cfg.para = 'whatever'
            cfg.tic = str(d)
            result = run_exps.main(cfg, remote=cfg.MUSCLE != 'local', gui=cfg.GUI)
            log('finished justrun with result: $', result)
            log('getting any results')

    if 'GETANDMAKE' in mode or (FULL and 'SAVE' in result):
        if mode != FULL:
            if cfg.MUSCLE != 'local':
                gcloud_config()
            result = 'SAVELOG'
        if 'SAVE' in result:
            getfigdata.main(result,TEMP_FIGS_FODLER,cfg)
            log("should call make figs next")
            # bin/makefigs overwrite
            from lib import makefigs
            makefigs.main(overwrite=False,root = TEMP_FIGS_FODLER)

        log('got result arg: ' + result)
    if FULL and 'SAVE' in result:
        File(TEMP_FIGS_FODLER).moveto(figsFolder)
        File(figsFolder).resolve('metadata.json').save(run_exps.exp_group_metadata)
        if FULL or 'COMPILE_TEST_ALL' in mode:
            from lib import compile_test_all
            compile_test_all.dnn(figsFolder)
        # shell('/Users/matt/miniconda3/bin/python3 bin/compile_test_all.py ' + figsFolder).interact()

        if FULL or 'MAKEREPORT' in mode:
            import makereport
            makereport.dnn()
        log('finished makereport')

# returns None if name was already used
FIGS_FOLDER = File('_figs/figs_dnn')
def get_figs_folder(nameRequest):
    all_names = []
    next_num = 0
    for f in FIGS_FOLDER.listmfiles():
        n, nam = tuple(f.name.split('-', 1))
        next_num = max(next_num, int(n))
        all_names += [nam]
    if nameRequest in all_names: return None
    return FIGS_FOLDER.resolve(f'{next_num + 1}-{nameRequest}')
if __name__ == '__main__':
    lib.boot.bootfun.register_exception_handler()
    kmscript('activate run tool window')

    prof = 'default'
    cfg = 'default'

    changes = {}

    if len(sys.argv) > 1:
        for a in sys.argv[1:]:
            if a.startswith('--'):
                k, v = tuple(a.replace('--', '').split('='))
                changes[k] = v
            elif a.startswith('-'):
                k, v = tuple(a.replace('-', '').split('='))
                if k == 'prof':
                    prof = v
                elif k == 'cfg':
                    cfg = v
                else:
                    err('arguments with one dash (-) need to be prof= or cfg=')
            else:
                err(f'invalid argument:{a} please see README')

    yml = File('cfg.yml').load()
    prof = yml['profiles'][prof]
    cfg = yml['configs'][cfg]

    if 'NTRAIN' in listkeys(cfg):
        prof_ntrain = prof['NTRAIN']
        for i, n in enum(cfg['NTRAIN']):
            if isstr(n) and n[0] == 'i':
                cfg['NTRAIN'][i] = prof_ntrain[int(n[1])]

    cfg = {**prof, **cfg}

    for k, v in listitems(changes):
        if k not in listkeys(cfg):
            err(f'invalid -- arguments: {k}, please see cfg.yml for configuration options')
        cfg[k] = v

    # hardcoded cfg
    cfg['root'] = '_figures'

    dnn(obj(cfg))
