from mlib.boot.lang import listkeys
from mlib.boot.stream import make2d, arr, concat
MET_PHASE = 'init'
FLAGS = None
eval_counter = 0

TEST_STEPS = None
step_counter = 1

EVAL_AND_REC_EVERY_EPOCH = False


reduced_map = {}
use_reduced_map = None

def met_phase(inc=False):
    if MET_PHASE is None: return None
    global eval_counter, TEST_STEPS, step_counter
    met_phase = MET_PHASE + str(step_counter)
    if inc:
        if 'fit' in MET_PHASE:
            step_counter = step_counter + 1
        elif 'eval' in MET_PHASE:
            eval_counter = eval_counter + 1
            if eval_counter == TEST_STEPS:
                step_counter = step_counter + 1
                eval_counter = 0
    return met_phase

GLOBAL_MET_LOG = dict()

CURRENT_PRED_MAP = None
CURRENT_TRUE_MAP = None

def num_pred_classes():
    return len(listkeys(CURRENT_PRED_MAP))

# num_classes=2
def reset_global_met_log():
    import lib.nn.net_mets as net_mets
    global eval_counter  # ,NUM_PRED_CLASSES
    # NUM_PRED_CLASSES = num_classes
    eval_counter = 0
    for m in net_mets.METS_TO_USE():
        GLOBAL_MET_LOG[str(m.__name__)] = make2d(arr(['init', -3]))

def update_met_log(fun, rrr, inc=False):
    phase = met_phase(inc)
    if phase is not None:
        GLOBAL_MET_LOG[fun.__name__] = concat(GLOBAL_MET_LOG[fun.__name__], [[phase, rrr]])
    return rrr
