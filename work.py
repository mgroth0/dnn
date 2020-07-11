import json

from mlib.boot.bootutil import margparse, setup_logging
from mlib.boot.mlog import setTic
from mlib.boot.mutil import isempty, prep_log_file
FLAGS = margparse(
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
    verbose=int,
    proto_model=int
)
from lib.boot import nn_init_fun
tf = nn_init_fun.setupTensorFlow(FLAGS)
if FLAGS.expid == '0' and (not FLAGS.gen) and (not FLAGS.deletenorms): nn_init_fun.NRC_IS_FINISHED()
setup_logging(verbose=FLAGS.verbose)
if FLAGS.tic is not None: setTic(FLAGS.tic * 1000)
prep_log_file('dnn/NRC', new=True)
from lib.nn_main import sym_net_main
from mlib.gpu import mygpus
FLAGS.mygpus = mygpus()
FLAGS.cfg_cfg = json.loads(FLAGS.cfg)
FLAGS.mygpufordata = FLAGS.mygpus[0] + 1 if not isempty(FLAGS.mygpus) else 1
from lib import data_saving
data_saving.root = FLAGS.cfg_cfg['root']
from lib.nn import nnstate
nnstate.FLAGS = FLAGS
nnstate.reset_global_met_log()
if float(1) == float(2):
    result_folder = nn_init_fun.runWithMultiProcess(sym_net_main)
else:
    result_folder = sym_net_main(FLAGS)
result_folder.zip_in_place()
result_folder.delete()
nn_init_fun.NRC_IS_FINISHED()
