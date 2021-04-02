import sys
from typing import Optional

from lib.decorators import FolderBuilder
from mlib.boot.mlog import progress





if __name__ == '__main__':
    # need to find way to automatically return
    from mdb import remote_breakpoint
    sys.breakpointhook = remote_breakpoint

    print('LETS_TRY_TO_HAVE_A_CLEAN_CONSOLE')

    import boot
    boot.take_tic_from_sysargv()
    from mlib.file import File
    # remote_portal = File('_remote_portal.pkl')
    import portal
    import pickle
    teleported = portal.PORTAL_FILE.load()
    # portal.CFG = teleported['CFG']
    globals().update(teleported['scope'])
    # with open('_remote_scope.pkl', 'rb') as scope:
    #     globals().update(pickle.load(scope))
    _boot_remote_result = None
    # with open('_remote_script.py', 'r') as pyscript:
    #     exec(pyscript.read().replace('return ', '_boot_remote_result='))


    def exec_pycode_and_return(pycode):
        exec(teleported['script'].replace('return ', '_dummy_var'))
        # noinspection PyUnresolvedReferences
        return _dummy_var

    if teleported['folder_builder']:
        fb = FolderBuilder()
        exec(teleported['script'])
        _boot_remote_result = fb.finish()
    else:
        exec(teleported['script'].replace('return ', '_boot_remote_result='))

    File('_boot_remote_result_folder').mkdirs()
    File('_boot_remote_result_folder').clear(silent=True)
    from mlib.boot.lang import isinstsafe
    if isinstsafe(_boot_remote_result, File):
        _boot_remote_result: Optional[File]
        _boot_remote_result.copy_into(File('_boot_remote_result_folder'))
    _boot_remote_result = None
    with open('_boot_remote_result.pkl', 'wb') as result:
        pickle.dump(
            _boot_remote_result,
            result,
            protocol=pickle.HIGHEST_PROTOCOL
        )
    boot.finish_dnn_remote()

# test change for rsync
