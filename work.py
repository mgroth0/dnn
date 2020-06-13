import importlib.util
import os
# INIT_FUN = f'{os.getcwd()}/src/main/python/lib/boot/initFun.py'
# INIT_FUN_NAME = INIT_FUN.split('/')[-1].split('.')[0]
# spec = importlib.util.spec_from_file_location(INIT_FUN_NAME, INIT_FUN)
# initFun = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(initFun)
from lib.boot import bootfun as initFun
FLAGS = initFun.margparse(
    cfg=str,
    para=str,
    tic=int,
    gpus=str,
    arch=str,
    epochs=int,
    gen=int,
    expid=str,
    ntrain=int,
    pipeline=str,
    batchsize=int,
    normtrainims=int,
    deletenorms=int,
    verbose=int
)
from lib.boot import nn_init_fun
tf = nn_init_fun.setupTensorFlow(FLAGS)
from lib.defaults import *
if FLAGS.expid == '0' and (not FLAGS.gen) and (not FLAGS.deletenorms): nn_init_fun.NRC_IS_FINISHED()
initFun.register_exception_handler()
initFun.setup_logging(verbose=FLAGS.verbose)
if FLAGS.tic is not None: setTic(FLAGS.tic * 1000)
prep_log_file('dnn/NRC', new=True)
from lib.nn_main import sym_net_main
if float(1) == float(2):
    result_folder = nn_init_fun.runWithMultiProcess(sym_net_main)
else:
    result_folder = sym_net_main(FLAGS)
result_folder.zip_in_place()
nn_init_fun.NRC_IS_FINISHED()
