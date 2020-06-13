from PyQt5.QtCore import Qt, pyqtBoundSignal
from PyQt5.QtGui import QFont

from lib.misc.guiutil import SimpleApp
import lib.run_exps_lib as run_exps_lib
from lib.run_exps_lib import *
from lib.defaults import *



@log_invokation
def main(cfg, remote=False, gui=True):
    if not remote:
        muscle = Muscle(num_gpus=0)
    exp_group_metadata = {}
    arch_md_strs = []
    archs = listmap(lambda e: e.arch, cfg.EXPS)
    if 'ALEX' in archs:
        arch_md_strs.append('\tALEX = AlexNet (pre-trained on ImageNet)')
    if 'GNET' in archs:
        arch_md_strs.append('\tGNET = GoogleNet (pre-trained on ImageNet)')
    if 'INC' in archs:
        arch_md_strs.append('\tINC = Inception ResNet V2 (pre-trained on ImageNet)')
    if 'SCRATCH' in archs:
        arch_md_strs.append('\tSCRATCH = ResNet18 (untrained)')

    exp_group_metadata['archs'] = arch_md_strs

    exp_group_metadata['nrepeats'] = cfg.REPEAT_ALL
    exp_group_metadata['nepochs'] = cfg.EPOCHS
    exp_group_metadata['batchsize'] = cfg.BATCH_SIZE
    exp_group_metadata['ntrainims'] = cfg.NTRAIN
    exp_group_metadata['normalized'] = cfg.NORM_TRAIN_IMS


    def pingChecker():
        f = File('_logs/local/pingchecker.log')
        f.deleteIfExists()
        f.mkparents()
        p = shell('ping www.google.com')
        while True:
            line = p.readline()
            # log('pingchecker got line')
            if len(line) == 0:
                log('pingchecker got EOF')
                f.append(f'({toc_str()})got EOF')
                break
            else:
                f.append(f'({toc_str()})' + utf_decode(line))



    run_in_daemon(pingChecker)

    jobs = [Job(
        {
            'para'        : cfg.para,
            'tic'         : cfg.tic,

            'expid'       : '0',

            'arch'        : 'JUST_CLEAR_FILES_AND_GEN_DATASET',
            'ntrain'      : 0,

            'pipeline'    : ''.replace(' ', ''),

            'epochs'      : cfg.EPOCHS,
            'batchsize'   : 0,
            'verbose'     : cfg.VERBOSE,
            'normtrainims': False,

            'deletenorms' : cfg.OVERWRITE_NORMS,
            'gen'         : cfg.REGEN_DATA  # implies kill
        },
        exp_cfg_o=obj({
            'gen_cfg': {
                'num_gpus'          : max(len(listkeys(muscle.GPU_IN_USE)), 2),
                'TRAINING_SET_SIZES': cfg.NTRAIN,
                'EVAL_SIZE'         : cfg.eval_nperc,
                'RSA_SIZE_PER_CLASS': cfg.rsa_nperc,
            },
            'root': cfg.root
        }),
        gpus=None,  # actually set to use all 4 below
        commands=[
            "rm -rf " + cfg.root,
            "find . -name \"*.pyc\" -exec rm -f {} \\;",
            "pkill -f miniconda3",
            "pkill -f MATLAB"
        ] if not remote else [],
        interact_with_nrc=cfg.INTERACT,
        remote=remote
    )]
    exp_id_file = File('_metastate.json')
    old = cfg.EXPS
    cfg.EXPS = []
    for i in range(cfg.REPEAT_ALL):
        for j in old:
            cfg.EXPS += [j]
    for e in cfg.EXPS:
        for ntrain in cfg.NTRAIN:
            exp_id = str(exp_id_file["next_exp_id"])
            exp_id_file["next_exp_id"] = int(exp_id) + 1
            jobs.append(Experiment(
                {
                    'para'        : cfg.para,
                    'tic'         : cfg.tic,

                    'expid'       : exp_id,

                    'arch'        : e.arch,
                    'ntrain'      : ntrain,

                    'pipeline'    : cfg.PIPELINE.replace(' ', ''),

                    'epochs'      : cfg.EPOCHS,
                    'batchsize'   : cfg.BATCH_SIZE,
                    'verbose'     : cfg.VERBOSE,
                    'normtrainims': cfg.NORM_TRAIN_IMS,

                    'deletenorms' : False,
                    'gen'         : False  # implies kill
                },
                exp_cfg_o=obj({
                    'root': cfg.root
                }),
                gpus=e.gpus,  # [0,1,2,3] if RUN_EXPS_IN_SERIAL else
                interact_with_nrc=cfg.INTERACT,
                remote=remote
            ))

    run_exps_lib.Job.TOTAL_TODO = len(jobs)

    muscle.runjob(jobs[0], listkeys(muscle.GPU_IN_USE))
    exps = jobs[1:]

    def all_done():
        r = True
        for exp in exps:
            if not exp.done:
                r = False
        return r

    if remote:
        statusP = google_compute.gc('', AUTO_LOGIN=True)
    else:
        statusP = ishell('bash')
    DONE_STR = 'DONEWITHCOMMAND'
    DONE_VAR = 'DONE_STR'
    MATT_STR = 'matt@'
    statusP.sendline(f'{DONE_VAR}={DONE_STR}')

    next_report = {'': 0}
    first_report = {'': True}
    REP_BAR_LENGTH = 100
    def report(signal, t):
        if not remote:
            if signal is not None:
                signal.emit('no local report yet')
            else:
                log('no local report yet')
            return
        report = '\n\n\t\t\t~~JOB REPORT~~'
        if first_report['']: signal.emit(report)
        for e in exps:
            report += f'\n{e.status()}'
            if first_report['']: signal.emit(report)
        report += '\n\n'
        if first_report['']: signal.emit(report)

        print(report)

        # clear buffer
        while True:
            line = statusP.readline_nonblocking(1)
            if line is None: break

        log('GETTING GPU REPORT')
        gpu_report = '\n\t\t\t~~GPU REPORT~~'
        statusP.sendline(f'nvidia-smi; echo ${DONE_VAR}')
        tesla_line = False
        percents = []
        while True:
            line = statusP.readline_nonblocking(1)
            if line is None or DONE_STR in line: break
            else:
                if tesla_line:
                    percents += [int(line.split('%')[0][-2:])]
                tesla_line = 'Tesla P4' in line
        for idx, perc in enum(percents):
            gpu_report += f'\n{idx}\t{insertZeros(perc, 2)}% {Progress.prog_bar(perc, BAR_LENGTH=REP_BAR_LENGTH)}'
        report += gpu_report
        if first_report['']: signal.emit(report)
        log('GETTING MEM REPORT')
        mem_report = '\n\n\t\t\t~~MEM REPORT~~'
        statusP.sendline(f'free -h; echo ${DONE_VAR}')
        log('send mem report request')
        while True:
            line = statusP.readline_nonblocking(1)
            if line is None or DONE_STR in line: break
            else:
                if MATT_STR not in line:
                    mem_report += f'\n{line}'
        report += mem_report
        if first_report['']: signal.emit(report)
        log('\nGETTING CPU REPORT')
        cpu_report = '\n\n\t\t\t~~CPU REPORT~~'
        # def trySendingCPURequest():
        statusP.sendline(
            f'''echo "CPU `LC_ALL=C top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\\1/" | awk '{{print 100 - $1}}'`% RAM `free -m | awk '/Mem:/ {{ printf("%3.1f%%", $3/$2*100) }}'` HDD `df -h / | awk '/\// {{print $(NF-1)}}'`"; echo ${DONE_VAR}''')
        log('SENT CPU LINE')
        report += cpu_report
        cpu_report = ''
        while True:
            line = statusP.readline_nonblocking(1)
            if line is None or DONE_STR in line: break
            else:
                if MATT_STR not in line:
                    cpu_report += f'\n{line}'

        cpu_stuff = tuple(listmap(lambda s: s.replace('%', ''), cpu_report.strip().split(' ')))
        if len(cpu_stuff) == 6:
            cpu_perc = float(cpu_stuff[1])
            ram_perc = float(cpu_stuff[3])
            hdd_perc = float(cpu_stuff[5])
            cpu_report = f'\nCPU\t{insertZeros(cpu_perc, 4)}% {Progress.prog_bar(cpu_perc, BAR_LENGTH=REP_BAR_LENGTH)}'
            cpu_report += f'\nRAM\t{insertZeros(ram_perc, 4)}% {Progress.prog_bar(ram_perc, BAR_LENGTH=REP_BAR_LENGTH)}'
            cpu_report += f'\nHDD\t{insertZeros(hdd_perc, 4)}% {Progress.prog_bar(hdd_perc, BAR_LENGTH=REP_BAR_LENGTH)}'

            report += cpu_report
        else:
            report += f'problem getting cpu_report ({len(cpu_stuff)=})'
        log('sending signal with REPORT')
        signal.emit(report)
        log('sent signal with REPORT')
        next_report[''] = t + 1
        first_report[''] = False

    def run_all_jobs(signal: pyqtBoundSignal):
        if signal is not None:
            signal.emit('Preparing to run all jobs...')
        else:
            log('Preparing to run all jobs...')
        t = None  # in case no exps
        while not all_done():
            t = time.time()
            if t >= next_report['']:
                report(signal, t)
            for e in exps:
                if not e.started:
                    if len(listkeys(muscle.GPU_IN_USE)) == 0:
                        muscle.runjob(e, [], a_sync=False)
                    else:
                        gpus_avail = []
                        for g in list(muscle.GPU_IN_USE.keys()):
                            if not muscle.GPU_IN_USE[g]:
                                gpus_avail.append(g)
                        if len(gpus_avail) >= e.gpus:
                            use_gpus = gpus_avail[0:e.gpus]
                            muscle.runjob(e, use_gpus, a_sync=not cfg.RUN_EXPS_IN_SERIAL)
                        else:
                            break
            time.sleep(0.1)
        report(signal, t)  # one last report when all are done

        [e.wait() for e in exps]  # wait 1

        if signal is not None:
            log('quiting app')
            signal.emit('QUIT MY APP')
        else:
            log('finished running all jobs')
        # app.quit()
    if gui:
        app = SimpleApp(
            sys.argv,
            title="GC Monitor",
            label="Stats:",
            background_fun=run_all_jobs
        )
        app.statsText = app.text('this should change (IF SERIAL IS TURNED OFF)')
        app.statsText.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        app.statsText.setFont(QFont("Monaco", 14))
        # @log_invokation
        def updateStatus(status):
            # log(f'updating status: {status}')
            if status == 'QUIT MY APP':
                app.quit()
            app.statsText.setText(status)
        app.update_fun = updateStatus
        app.exec()
        activateIdea()
    else:
        run_all_jobs(None)

    [e.wait() for e in exps]  # wait 2, after app exits. Should not be needed.
    Job.kill_all_jobs()  # should not be needed, but just in case

    # File('exp_group_name.txt').write(EXP_GROUP_NAME)
    rr = ''
    if cfg.SAVE_DATA: rr = rr + 'SAVE'
    if cfg.GET_LOGS: rr = rr + 'LOG'

    return rr
