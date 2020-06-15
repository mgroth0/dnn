#!/Users/matt/miniconda3/bin/python3
# noinspection PyUnresolvedReferences
from glob import glob

from lib.misc.google_compute import gc
from lib.defaults import *

def main(arg, root,cfg):
    if cfg.MUSCLE != 'local':
        log('running getfigdata')
        if 'SAVE' in arg:
            mkdir(root)
            # set timeout -1
            # spawn bash gcloud compute  --project "neat-beaker-261120" scp test-3:~/mitili/{root} . --recurse
            p = gc('', 'get', f'/home/matt/mitili/{root}/*.zip', root, RECURSE=True, AUTO_LOGIN=True)
            p.interact()

        if 'LOG' in arg:
            p = gc('', 'get', '/home/matt/mitili/_logs/remote/dnn', '_logs/remote', RECURSE=True, AUTO_LOGIN=True)
            p.interact()

    # import sys
    # sys.exit()
    if 'SAVE' in arg:
        log('unzipping')
        for f in glob(f'{root}/*.zip'):
            shell(f'unzip -o {f} -d {root}').readlines()

        if cfg.MUSCLE != 'local':


            for f in glob(f'figures2/home/matt/mitili/{root}/*'):
                shell(f'mv {f} {root}').interact()

            shell(f'rm -rf {f}/home').interact()

        log('unzipped')

        for f in glob(f'{root}/*.zip'):
            shell(f'rm {f}').interact()
        # shell(f'rm -rf {root}/0').interact()
