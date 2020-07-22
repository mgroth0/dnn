import sys

from mlib.boot import log
from mlib.file import File, abspath
from mlib.shell import SSHExpectProcess, InteractiveExpectShell
from mlib.str import utf_decode
from mlib.term import log_invokation
PROJECT_NAME = 'test-3'
PROJECT = ['--project', 'neat-beaker-261120', '--zone', 'us-central1-a', PROJECT_NAME]
def isrunning():
    return 'RUNNING' in gc('', 'list').all_output()
def gc(*args, AUTO_LOGIN=False, RECURSE=False):
    SSH = len(args) <= 1
    arg = 'ssh' if SSH else args[1]
    STOPIN = arg == 'stopin'
    if STOPIN:
        SSH = True
        arg = 'ssh'
    STOP = arg == 'stop'
    START = arg == 'start'
    LIST = arg == 'list'
    PUT = arg == 'put'
    GET = arg == 'get'
    if PUT or GET: arg = 'scp'
    COMPUTE = ['/Users/matt/google-cloud-sdk/bin/gcloud', 'compute']
    if STOP or START or LIST:
        COMPUTE += ['instances']
    COMMAND = COMPUTE + [arg]
    if STOP or START or SSH:
        COMMAND += PROJECT
    if PUT or GET:
        FROM = ((PROJECT_NAME + ':') if GET else '') + abspath(args[2], remote=GET)
        TO = ((PROJECT_NAME + ':') if PUT else '') + abspath(args[3], remote=PUT)
        if File(FROM).isdir() or RECURSE:
            COMMAND.append('--recurse')
        COMMAND.extend([FROM, TO])
    if SSH:
        COMMAND.append('--ssh-flag="-vvv"')
        p = GCShell(COMMAND)
        if STOPIN:
            p.login()
            if args[2] == 'w':
                p.sendline(
                    './stopinw'
                )
                p.readline() # input line
                w = utf_decode(p.readline())
                if '1969' in w:
                    log('no shutdown is scheduled')
                else:
                    log(f'shutdown is scheduled for {w}')
            elif args[2] == 'c':
                p.sudo(['shutdown', '-c'])
                log('cancelled shutdown')
            else:
                p.sudo(['shutdown', '-h', args[2]])

                log(f'scheduled shutdown for {args[2]} mins')
            p.close()
            return None
    else:
        p = GCProcess(COMMAND)
    if AUTO_LOGIN: p.login()
    return p


class GCProcess(SSHExpectProcess): pass
class GCShell(GCProcess, InteractiveExpectShell):
    def login(self):
        super().login()
        self.expect("matt@test-3")
    @log_invokation(with_instance=True)
    def restart(self):
        log('this function probably wont fix the issue if the process is blocking elsewhere')
        self.p.close()
        log('closed p')
        self.p = self._start()

def gcloud_config():
    # I think, but not sure, that this doesn't have to be run every single time the instance restarts but rather every time it changes ip or something. like once a week or something? idk
    # this solved a problem I was having where rsync was throwing an error
    # https://stackoverflow.com/questions/27857532/rsync-to-google-compute-engine-instance-from-jenkins
    child = SSHExpectProcess('/Users/matt/google-cloud-sdk/bin/gcloud compute config-ssh', timeout=None,
                             logfile_read=sys.stdout.buffer)
    r = child.expect(['You should', "passphrase"])

    if r == 1:
        child.sendpass()

    # this has to be called or it will block
    child.readlines()

    child.wait()
