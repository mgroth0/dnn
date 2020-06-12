import lib.boot
from lib.boot.bootfun import margparse
from lib.defaults import *
from lib.gui import answer_request
from lib.misc.google_compute import gcloud_config


def main(
        FLAGS
):
    mode = FLAGS.mode
    FULL = 'FULL' in mode
    log('starting pushr')
    if 'CLEAN' in mode or FULL:
        shell('pkill -f miniconda3')
        shell('pkill -f MATLAB')
    import exec.run_exps as run_exps
    if FULL and run_exps.EXP_MOD.SAVE_DATA:
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
                                                   gui=FLAGS.gui)
        metastate["last_submitted_exp_group_name"] = figsFolder
        figsFolder = File(figsFolder).abspath

        if not figsFolder:
            log('did not get a fig folder')
            exit(1)
        log('figsFolder=$', figsFolder)
    if 'CLEAN' in mode or FULL:
        log('cleaning')
        File('figures2').deleteIfExists()
        File('_logs/local/dnn').deleteIfExists()
        File('_logs/remote/dnn').deleteIfExists()
    if FULL or 'JUSTRUN' in mode or 'PUSH' in mode:
        log('about to run justrun')
        log('in justrun')
        d = int(time.time())
        # File(pwd()+'/bin/tic.txt').write(d)
        log('pushing')

        if FLAGS.muscle != 'LOCAL':
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
            result = run_exps.main([sys.argv[1], str(d)], remote=FLAGS.muscle != 'LOCAL',gui=FLAGS.gui)
            log('finished justrun with result: $', result)
            log('getting any results')

    if 'GETANDMAKE' in mode or (FULL and 'SAVE' in result):
        if mode != FULL:
            gcloud_config()
            result = 'SAVELOG'
        if 'SAVE' in result:
            log("should call make figs next")
            # bin/makefigs overwrite
            from lib import makefigs
            makefigs.main(overwrite=False)

        log('got result arg: ' + result)
    log('between getandmake and compile_test_all')
    if FULL and 'SAVE' in result:
        File('figures2').moveto(figsFolder)
        File(figsFolder).resolve('metadata.json').save(run_exps.exp_group_metadata)
        if FULL or 'COMPILE_TEST_ALL' in mode:
            import compile_test_all
            compile_test_all.main(figsFolder)
        # shell('/Users/matt/miniconda3/bin/python3 bin/compile_test_all.py ' + figsFolder).interact()

        if FULL or 'MAKEREPORT' in mode:
            import makereport
            makereport.main()
        log('finished makereport')

    log('finished pushr')
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
    main(margparse(
        mode=str,
        muscle=str,
        gui=int
        # FULL=int,
        # CLEAN=int,
        # JUSTRUN=int,
        # GETANDMAKE=int,
        # MAKEREPORT=int
    ))
