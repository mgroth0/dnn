import time

from PyQt5.QtCore import pyqtBoundSignal

from lib.dnn_jobs import DNN_Job
from lib.dnn_proj_struct import DNN_ExperimentGroup
from lib.misc import google_compute
from lib.status_reporter import StatusReporter
from mlib.boot import log
from mlib.boot.lang import listkeys
from mlib.job import Job
from mlib.term import log_invokation
class Muscle:
    def __init__(self, local, num_gpus=0):
        self.local = local
        self.GPU_IN_USE = {}
        for i in range(num_gpus):
            self.GPU_IN_USE[i] = False
    def runjob(self, j: DNN_Job, gpus_to_use, a_sync=False):
        for g in gpus_to_use: self.GPU_IN_USE[g] = True
        j.run(gpus_to_use, self, a_sync=a_sync)


    def run_all_jobs_main(self, jobs, serial, gui):
        Job.TOTAL_TODO = len(jobs)

        self.runjob(jobs[0], listkeys(self.GPU_IN_USE))

        exps = jobs[1:]

        def all_done():
            r = True
            for expe in exps:
                if not expe.done:
                    r = False
            return r

        status_reporter = StatusReporter(remote=not self.local)

        @log_invokation
        def run_all_jobs(signal: pyqtBoundSignal):
            if signal is not None:
                signal.emit('Preparing to run all jobs...')
            t = None  # in case no exps
            while not all_done():
                t = time.time()
                if t >= status_reporter.next_report['']:
                    status_reporter.report(signal, t, exps)
                for e in exps:
                    if not e.started:
                        if len(listkeys(self.GPU_IN_USE)) == 0:
                            self.runjob(e, [], a_sync=False)
                        else:
                            gpus_avail = []
                            for g in list(self.GPU_IN_USE.keys()):
                                if not self.GPU_IN_USE[g]:
                                    gpus_avail.append(g)
                            if len(gpus_avail) >= e.gpus:
                                use_gpus = gpus_avail[0:e.gpus]
                                self.runjob(e, use_gpus, a_sync=not serial)
                            else:
                                break
                time.sleep(0.1)
            status_reporter.report(signal, t, exps)

            [ex.wait() for ex in exps]  # wait 1

            if signal is not None:
                log('quiting app')
                signal.emit('QUIT MY APP')



        if gui:
            status_reporter.run_app(function=run_all_jobs)
        else:
            run_all_jobs(None)

    @log_invokation()
    def pull_data(self, TEMP_FIGS_FOLDER, cfg, new_fig_folder):
        if not self.local:  # google cloud
            google_compute.gcloud_config()
            if cfg.SAVE_DATA:
                TEMP_FIGS_FOLDER.mkdir()
                google_compute.gc('', 'get', f'/home/matt/mitili/{TEMP_FIGS_FOLDER.abspath}/*.zip',
                                  TEMP_FIGS_FOLDER.abspath, RECURSE=True,
                                  AUTO_LOGIN=True).interact()
            if cfg.LOG_DATA:
                google_compute.gc('', 'get', '/home/matt/mitili/_logs/remote/dnn', '_logs/remote', RECURSE=True,
                                  AUTO_LOGIN=True).interact()
        [f.unzip_to(TEMP_FIGS_FOLDER, delete_after=True) for f in TEMP_FIGS_FOLDER.glob(f'*.zip')]
        TEMP_FIGS_FOLDER.moveto(new_fig_folder)
        return DNN_ExperimentGroup.from_folder(new_fig_folder)
