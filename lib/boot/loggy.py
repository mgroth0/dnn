import os
import time

from ..mynvidia import gpu_mem_str
from ..vals import USE_THREADING
ticTime = None
LOG_FILE = None
STATUS = dict()
import numpy as np
import sys

def initTic():
    # err('')
    if ticTime is None:
        with open('bin/tic.txt', 'r') as myfile:
            data = myfile.read().replace('\n', '')

        setTic(np.long(data) * 1000)

        if len(sys.argv) > 2 and sys.argv[2] == 'tic':
            setTic(time.time() * 1000)

        log('got tic')

def getNextIncrementalFile(file):
    import lib.misc.mutil as mutil
    file = mutil.File(file)
    onename = file.name.replace('.', '_1.')
    onefile = mutil.File(file.parentDir).resolve(onename)
    if not onefile.exists():
        return onefile
    else:
        if '_' in file.name:
            base = file.name.split('_')[0]
            ext = file.name.split('_')[1].split('.')[1]
            n = int(file.name.split('_')[1].split('.')[0])
            n = n + 1
        else:
            base = file.name.split('.')[0]
            ext = file.name.split('.')[1]
            n = 1
        return mutil.File(file.parentDir).resolve(base + '_' + str(n) + '.' + ext)

def prep_log_file(filename, new=False):
    if filename is None:
        filename = os.path.basename(sys.argv[0]).replace('.py', '')

    import lib.misc.mutil as mutil
    import lib.boot.bootutil as bootutil
    if bootutil.ismac():
        filename = '_logs/local/' + filename
    else:
        filename = '_logs/remote/' + filename

    filename = filename + '.log'
    filename = mutil.MITILI_FOLDER().respath(filename)

    if new:
        filename = getNextIncrementalFile(filename)

    import lib.misc.mutil as mutil
    global LOG_FILE
    if LOG_FILE is None:
        LOG_FILE = mutil.File(filename)
    if LOG_FILE.exists():
        LOG_FILE.delete()
    LOG_FILE.mkparents()
    LOG_FILE.touch()
    print('initialized LOG_FILE:' + str(LOG_FILE))

def setTic(t):
    global ticTime
    ticTime = t

def tic():
    global ticTime
    ticTime = time.time() * 1000
    return ticTime

TIC = tic()

def toc():
    global ticTime
    if ticTime is None:
        # print('NOT GOOD, ticTIME  is None (' + s + ')')
        return -1
        # err('not good')
        # tic()
    return (time.time() * 1000) - ticTime

def resize_str(s, n):
    if len(s) > n:
        s = s[0:n]
    elif len(s) < n:
        dif = n - len(s)
        for i in range(0, dif):
            s = s + ' '
    return s

def toc_str(t=None):
    if t is None:
        return f'{toc() / 1000:.2f}'
    else:
        return f'{t:.2f}'

STARTED_GPU_INFO_THREAD = False
from multiprocessing import Process, Queue
gpu_q = None
gpu_q_stop = Queue(maxsize=1)
import atexit
def stop_gpu_info_fun():
    gpu_q_stop.put('anything')
atexit.register(stop_gpu_info_fun)
GPU_WATCH_PERIOD_SECS = 1
latest_gpu_str = ''

def gpu_info_fun(gpu_q, GPU_WATCH_PERIOD_SECS):
    while gpu_q_stop.empty():
        s = gpu_mem_str()
        gpu_q.put(s, block=True)
        time.sleep(GPU_WATCH_PERIOD_SECS)

import platform
mac = platform.system() == 'Darwin'




def get_log_info(ss, *args):
    global STARTED_GPU_INFO_THREAD, gpu_q, latest_gpu_str
    if not mac and not STARTED_GPU_INFO_THREAD:
        STARTED_GPU_INFO_THREAD = True
        gpu_q = Queue(maxsize=1)
        p = Process(target=gpu_info_fun, args=(gpu_q, GPU_WATCH_PERIOD_SECS))
        p.start()
        # p.join()

    t_start = time.time()

    # print('log time 1: ' + str(time.time() - t_start))

    if LOG_FILE is None:
        print('auto-prepping log file')
        prep_log_file(None)

    # print('log time 2: ' + str(time.time() - t_start))

    t = toc() / 1000
    v = toc_str(t)
    import os.path
    import traceback
    old_s = str(ss)

    # print('log time 3: ' + str(time.time() - t_start))

    for idx, aa in enumerate(args):
        ss = ss.replace("$", str(aa), 1)

    # print('log time 4: ' + str(time.time() - t_start))

    # print('might log: ' + str(vals.IN_SERIAL_MODE))
    # try:

    stack = traceback.extract_stack()

    # print('log time 5: ' + str(time.time() - t_start))

    if len(stack) == 1:
        file = 'MATLAB'
    else:
        file = os.path.basename(stack[-3][0]).split('.')[0]

    # print('log time 6: ' + str(time.time() - t_start))
    # except:
    #     print('weird thing happened')
    #     print(traceback.extract_stack())
    #     print(traceback.extract_stack()[-2])
    #     print(traceback.extract_stack()[-2][0])
    #     print(traceback.extract_stack()[-2][0].split('.'))
    #     print(os.path.basename(traceback.extract_stack()[-2][0]).split('.')[0])
    # if vals.IN_SERIAL_MODE or vvars.i == 0 :


    line_start = '[' + processTag() + '|' + v + '|'

    # print('log time 7: ' + str(time.time() - t_start))

    if not mac:
        if not gpu_q.empty():
            latest_gpu_str = gpu_q.get()
        line_start = line_start + latest_gpu_str + '|'

    # print('log time 8: ' + str(time.time() - t_start))

    line = line_start + resize_str(file, 14) + '] ' + str(ss)

    # file_line = line_start + resize_str(file, 10) + ') ' + ss
    # why was I using old_s for file_line??? for the LogViewer!
    file_line = line_start + resize_str(file, 10) + ') ' + old_s

    # print('log time 9: ' + str(time.time() - t_start))

    return line, file_line, t

def log(ss, *args, silent=False):
    line, file_line, v = get_log_info(ss, *args)

    if not silent:
        print(line)

    # print('log time 10: ' + str(time.time() - t_start))

    with open(LOG_FILE.abspath, "a") as myfile:
        myfile.write(file_line + "\n")

    # print('log time 11: ' + str(time.time() - t_start))

    return v

def processTag():
    name = processName()
    if USE_THREADING:
        return name
    else:
        if 'Main' in name:
            return 'MP'
        else:
            return 'L' + name[-1]

thread_name_dict = dict()

def processName():
    if USE_THREADING:
        import threading
        ident = threading.get_ident()
        if ident in thread_name_dict.keys():
            name = thread_name_dict[ident]
        else:
            name = len(thread_name_dict.keys()) + 1
            thread_name_dict[ident] = name
        return 'T' + str(name)
    else:
        import multiprocessing
        return multiprocessing.current_process().name

from . import nn_init_fun
def nrc_is_finished_upgrade(old):
    def f():
        log('NRC_IS_FINISHED')
        old()
    return f
if nn_init_fun.NRC_IS_FINISHED.__name__ != 'f':
    nn_init_fun.NRC_IS_FINISHED = nrc_is_finished_upgrade(nn_init_fun.NRC_IS_FINISHED)
