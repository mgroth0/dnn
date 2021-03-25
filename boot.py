def take_tic_from_sysargv():
    import sys
    for s in sys.argv:
        if '--tic=' in s:
            import mlib.boot.mlog
            mlib.boot.mlog.setTic(float(s.replace('--tic=', '')))

def finish_dnn_remote():
    from mlib.file import File
    from mlib.boot.crunch import PIPELINE_SECTION_FILE
    import mlib.boot.crunch
    File(PIPELINE_SECTION_FILE).save(mlib.boot.crunch.PIPELINE_SECTIONS)
    print('printing dnn is finished string!')
    print('__DNN_IS_FINISHED__')

if __name__ == '__main__':
    take_tic_from_sysargv()
    # noinspection PyUnresolvedReferences
    import dnn
