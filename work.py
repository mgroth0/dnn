import json

from mlib.boot.mlog import setTic, setup_logging
from mlib.input import margparse

def main(FLAGS):
    from lib.boot import nn_init_fun
    tf = nn_init_fun.setupTensorFlow(FLAGS)
    if FLAGS.expid == '0' and (not FLAGS.gen) and (not FLAGS.deletenorms): nn_init_fun.NRC_IS_FINISHED()
    setup_logging(verbose=FLAGS.verbose)
    if FLAGS.tic is not None:
        setTic(int(FLAGS.tic) * 1000)
    from mlib.proj.struct import Project
    Project.prep_log_file('dnn/NRC', new=True)
    from lib.nn_main import nnet_main
    from mlib.gpu import mygpus
    FLAGS.mygpus = mygpus()
    if isinstance(FLAGS.cfg, str):
        FLAGS.cfg_cfg = json.loads(FLAGS.cfg)
    else:
        FLAGS.cfg_cfg = FLAGS.cfg.toDict()
    FLAGS.mygpufordata = 1  # because I am not managing GPUs on OpenMind in the same way
    # FLAGS.mygpufordata = FLAGS.mygpus[0] + 1 if not isempty(FLAGS.mygpus) else 1
    from lib import dnn_data_saving
    dnn_data_saving.root = FLAGS.cfg_cfg['root']
    from lib.nn import nnstate
    nnstate.FLAGS = FLAGS
    import lib.nn.net_mets as net_mets
    if FLAGS.salience:
        net_mets._mets_to_use.remove(net_mets.mcc_multi)
    nnstate.reset_global_met_log()
    if float(1) == float(2):
        result_folder = nn_init_fun.runWithMultiProcess(nnet_main)
    else:
        result_folder = nnet_main(FLAGS)
    result_folder.zip_in_place()
    result_folder.delete()
    nn_init_fun.NRC_IS_FINISHED()


if __name__ == '__main__':
    FLAGS = margparse(
        cfg=str,
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
        proto_model=int,
        salience=int,
        TRANSFER_LEARNING=int,
        REGEN_NTRAIN=int,
        PRED_SIZE=int
    )
