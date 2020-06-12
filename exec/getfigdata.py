#!/Users/matt/miniconda3/bin/python3
# noinspection PyUnresolvedReferences
from glob import glob

import import_mod
from google_compute import gc
from lib.defaults import *

def main(arg):
    log('running getfigdata')
    if 'SAVE' in arg:
        mkdir('figures2')
        # set timeout -1
        # spawn bash gcloud compute  --project "neat-beaker-261120" scp test-3:~/mitili/figures2 . --recurse
        p = gc('', 'get', '/home/matt/mitili/figures2/*.zip', 'figures2',RECURSE=True,AUTO_LOGIN=True)
        p.interact()

    if 'LOG' in arg:
        p = gc('', 'get', '/home/matt/mitili/_logs/remote/dnn', '_logs/remote',RECURSE=True,AUTO_LOGIN=True)
        p.interact()

    # import sys
    # sys.exit()
    if 'SAVE' in arg:
        log('unzipping')
        for f in glob('figures2/*.zip'):
            shell(f'unzip -o {f} -d figures2').readlines()

        for f in glob('figures2/home/matt/mitili/figures2/*'):
            shell(f'mv {f} figures2').interact()

        shell('rm -rf figures2/home').interact()

        log('unzipped')

        for f in glob('figures2/*.zip'):
            shell('rm ' + f).interact()
        shell('rm -rf figures2/0').interact()

if __name__ == '__main__':
    main('SAVELOG')
