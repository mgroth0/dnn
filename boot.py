def take_tic_from_sysargv():
    import sys
    for s in sys.argv:
        if '--tic=' in s:
            import mlib.boot.mlog
            mlib.boot.mlog.setTic(float(s.replace('--tic=', '')))

def finish_dnn_remote():
    from mlib.file import File
    from mlib.boot.crunch import get_manager, PIPELINE_SECTION_FILE
    File(PIPELINE_SECTION_FILE).save(get_manager().PIPELINE_SECTIONS, silent=True)
    from mlib.boot import info
    info('printing dnn is finished string!')
    print('__DNN_IS_FINISHED__')

if __name__ == '__main__':
    take_tic_from_sysargv()
    # noinspection PyUnresolvedReferences
    import dnn
