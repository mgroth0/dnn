# noinspection PyUnresolvedReferences

from lib.figs.JsonSerializable import obj
import lib.misc.google_compute as google_compute
# import intercept
# intercept.MULTI_INTERCEPT = True
from lib.defaults import *




class Job:
    ALL_Ps = []
    TOTAL_FINISHED = 0
    TOTAL_TODO = None
    next_instance_mindex = 1
    def __init__(self,
                 exp_args: dict,
                 exp_cfg_o,
                 gpus=(),
                 commands=(),
                 interact_with_nrc=False,
                 remote=False
                 ):
        self.remote = remote
        self.exp_cfg_j = json.dumps(exp_cfg_o.toDict()).replace('"', '\\"')
        self.commands = commands

        self.interact_with_nrc = interact_with_nrc

        self.exp_args = exp_args

        self.gpus = gpus

        self.regen = exp_args['gen']  # implies kill
        self.deletenorms = exp_args['deletenorms']  # implies kill

        # for Job Report
        self.arch = exp_args['arch']
        self.ntrain = exp_args['ntrain']

        assert not (self.regen and self.deletenorms)

        self.main_args = [f'--{k}={arg_str(v)}' for k, v in exp_args.items()]

        self.instance_idx = Job.next_instance_mindex
        Job.next_instance_mindex += 1

        self.started = False
        self.start_time = None
        self.done = False
        self.end_time = None
        self.using_gpus = [None]

        self.last_log = None





    def __str__(self): return f'Job {self.instance_idx}'



    class JobStatus:
        def __init__(self, job):
            status = 'PENDING '
            if job.started:
                status = 'RUNNING '
            if job.done:
                status = 'FINISHED'
            self.job = job
            self.status = status
            self.gpu = job.using_gpus[0]
            self.run_time_str = job.run_time_str()
            self.last = shorten_str(job.last_log,
                                    # 48
                                    inf
                                    # .replace('\n', '/n')
                                    ).strip().replace('\r', '/r')
        def __str__(self):
            return f'{self.job}\t{self.status}\tGPU={self.gpu}\t{self.run_time_str}\t\t{self.last}'

    def status(self): return Job.JobStatus(self)
    @log_instance_invokation
    def run(self, gpus_to_use, muscle, a_sync=False):
        self.started = True
        self.start_time = time.time()
        self.using_gpus = gpus_to_use
        nrc_args = [f'--gpus={"".join(strs(gpus_to_use))}'] + self.main_args
        nrc_args += [f"--cfg=\"\"\"{self.exp_cfg_j}\"\"\""]
        if self.remote:
            p = google_compute.gc(AUTO_LOGIN=True)
            p.cd("~/mitili")
        else:
            p = ishell('bash')
            # class relproc(Process):
            #     def run(self):
            #         import exec.new_result_comp as nrc
            #
            # p = relproc()
        Job.ALL_Ps += [p]
        for com in self.commands: p.sendline(com)
        if self.remote:
            p: InteractiveShell
            p.py(f'src/main/python/exec/work.py {" ".join(nrc_args)}')
        else:
            # p: Process
            from os.path import expanduser
            home = expanduser("~")
            p.sendline(f'{home}/miniconda3/envs/dnn/bin/python work.py {" ".join(nrc_args)}')
            # p.start()
        if self.interact_with_nrc: p.interact()
        else:
            string_holder = {'': ''}
            def save_last_log(data, o):
                data = utf_decode(data)
                string_holder[''] += ''
                if not isblank(data):
                    for line in reversed(data.split('\n')):
                        if not isblank(line):
                            o.last_log = line
                            break
            p.log_to_stdout(fun=save_last_log, o=self)
        if a_sync:
            run_in_thread(self.inter_p_wrap, args=(p, gpus_to_use, muscle))
        else:
            self.inter_p_wrap(p, gpus_to_use, muscle)


    def run_time_str(self):
        if self.start_time is None:
            return '...'
        elif self.end_time is None:
            return min_sec_form(time.time() - self.start_time) + '...'
        else:
            return min_sec_form(self.end_time - self.start_time)

    def inter_p_wrap(self, p, gpus_im_using, muscle):
        if not self.interact_with_nrc:
            log('waiting for child...')
            r = p.expect(["NRC IS FINISHED", pexpect.EOF, 'ERROR ERROR ERROR'])
            log({
                    0: 'run_exps got a success',
                    1: 'run_exps got an EOF... what? exiting run_exps',
                    2: 'run_exps got an error, exiting run_exps'
                }[r])
            p.close()
            log('closed child')
            if r in [1, 2]: self.explode()
        Job.TOTAL_FINISHED += 1
        print(f'Finished Job {self} ({Job.TOTAL_FINISHED}/{Job.TOTAL_TODO})')
        self.done = True
        self.end_time = time.time()
        for g in gpus_im_using: muscle.GPU_IN_USE[g] = False

    def wait(self):
        while not self.done:
            time.sleep(0.1)

    def explode(self):
        self.kill_all_jobs()
        import os
        os._exit(0)

    @classmethod
    @log_invokation
    def kill_all_jobs(cls):
        for p in cls.ALL_Ps:
            if p.alive(): p.close()


class Experiment(Job):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expid = self.exp_args['expid']
    def __str__(self):
        arch_str = f'ARCH={self.arch}{self.ntrain}'
        arch_str = lengthen_str(arch_str, 20)
        return f'{super().__str__()}\tEXPID={self.expid}\t{arch_str}'


class Muscle:
    def __init__(self, num_gpus=0):
        self.GPU_IN_USE = {}
        for i in range(num_gpus):
            self.GPU_IN_USE[i] = False
    def runjob(self, j: Job, gpus_to_use, a_sync=False):
        for g in gpus_to_use: self.GPU_IN_USE[g] = True
        j.run(gpus_to_use, self, a_sync=a_sync)
