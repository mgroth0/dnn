from dataclasses import dataclass
from typing import ClassVar

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from lib.misc import google_compute
from mlib.boot import log
from mlib.boot.lang import enum
from mlib.boot.stream import listmap, insertZeros
from mlib.guiutil import SimpleApp
from mlib.km import activateIdea
from mlib.shell import ishell
from mlib.term import Progress


@dataclass
class StatusReporter:
    DONE_STR: ClassVar[str] = 'DONEWITHCOMMAND'
    DONE_VAR: ClassVar[str] = 'DONE_STR'
    MATT_STR: ClassVar[str] = 'matt@'
    remote: bool
    def __post_init__(self):
        if self.remote:
            self.statusP = google_compute.gc('', AUTO_LOGIN=True)
        else:
            self.statusP = ishell('bash')
        self.statusP.sendline(f'{self.DONE_VAR}={self.DONE_STR}')
        self.next_report = {'': 0}
        self.first_report = {'': True}
        self.REP_BAR_LENGTH = 100
    def report(self, signal, t, exps):
        if not self.remote:
            if signal is not None:
                signal.emit('no local report yet')
            else:
                log('no local report yet')
            return
        the_report = '\n\n\t\t\t~~JOB REPORT~~'
        if self.first_report['']: signal.emit(the_report)
        for e in exps:
            the_report += f'\n{e.status()}'
            if self.first_report['']: signal.emit(the_report)
        the_report += '\n\n'
        if self.first_report['']: signal.emit(the_report)

        print(the_report)

        while True:  # clear buffer
            line = self.statusP.readline_nonblocking(1)
            if line is None: break

        log('GETTING GPU REPORT')
        gpu_report = '\n\t\t\t~~GPU REPORT~~'
        self.statusP.sendline(f'nvidia-smi; echo ${self.DONE_VAR}')
        tesla_line = False
        percents = []
        while True:
            line = self.statusP.readline_nonblocking(1)
            if line is None or self.DONE_STR in line: break
            else:
                if tesla_line:
                    percents += [int(line.split('%')[0][-2:])]
                tesla_line = 'Tesla P4' in line
        for idx, perc in enum(percents):
            gpu_report += f'\n{idx}\t{insertZeros(perc, 2)}% {Progress.prog_bar(perc, BAR_LENGTH=self.REP_BAR_LENGTH)}'
        the_report += gpu_report
        if self.first_report['']: signal.emit(the_report)
        log('GETTING MEM REPORT')
        mem_report = '\n\n\t\t\t~~MEM REPORT~~'
        self.statusP.sendline(f'free -h; echo ${self.DONE_VAR}')
        log('send mem report request')
        while True:
            line = self.statusP.readline_nonblocking(1)
            if line is None or self.DONE_STR in line: break
            else:
                if self.MATT_STR not in line:
                    mem_report += f'\n{line}'
        the_report += mem_report
        if self.first_report['']: signal.emit(the_report)
        log('\nGETTING CPU REPORT')
        cpu_report = '\n\n\t\t\t~~CPU REPORT~~'
        self.statusP.sendline(
            f'''echo "CPU `LC_ALL=C top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\\1/" | awk '{{print 100 - $1}}'`% RAM `free -m | awk '/Mem:/ {{ printf("%3.1f%%", $3/$2*100) }}'` HDD `df -h / | awk '/\// {{print $(NF-1)}}'`"; echo ${self.DONE_VAR}''')
        log('SENT CPU LINE')
        the_report += cpu_report
        cpu_report = ''
        while True:
            line = self.statusP.readline_nonblocking(1)
            if line is None or self.DONE_STR in line: break
            else:
                if self.MATT_STR not in line:
                    cpu_report += f'\n{line}'

        cpu_stuff = tuple(listmap(lambda s: s.replace('%', ''), cpu_report.strip().split(' ')))
        if len(cpu_stuff) == 6:
            cpu_perc = float(cpu_stuff[1])
            ram_perc = float(cpu_stuff[3])
            hdd_perc = float(cpu_stuff[5])
            cpu_report = f'\nCPU\t{insertZeros(cpu_perc, 4)}% {Progress.prog_bar(cpu_perc, BAR_LENGTH=self.REP_BAR_LENGTH)}'
            cpu_report += f'\nRAM\t{insertZeros(ram_perc, 4)}% {Progress.prog_bar(ram_perc, BAR_LENGTH=self.REP_BAR_LENGTH)}'
            cpu_report += f'\nHDD\t{insertZeros(hdd_perc, 4)}% {Progress.prog_bar(hdd_perc, BAR_LENGTH=self.REP_BAR_LENGTH)}'

            the_report += cpu_report
        else:
            the_report += f'problem getting cpu_report ({len(cpu_stuff)=})'
        log('sending signal with REPORT')
        signal.emit(the_report)
        log('sent signal with REPORT')
        self.next_report[''] = t + 1
        self.first_report[''] = False

    @staticmethod
    def run_app(function):
        app = SimpleApp(
            sys.argv,
            title="GC Monitor",
            label="Stats:",
            background_fun=function
        )
        app.statsText = app.text('this should change (IF SERIAL IS TURNED OFF)')
        app.statsText.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        app.statsText.setFont(QFont("Monaco", 14))
        def updateStatus(status):
            if status == 'QUIT MY APP':
                app.quit()
            app.statsText.setText(status)
        app.update_fun = updateStatus
        app.exec()
        activateIdea()
