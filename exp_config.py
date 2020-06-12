from lib.figs.JsonSerializable import obj

def selected_config():
    return [
        # GEN_IMAGES
        PRE_PRE_PREPROCESS
        # TEST_ONE
        # TEST_FOUR
        # DEFAULT_CONFIG
        # DEFAULT_NO_NORM
    ][0]



def RunExpsConfig(
        RUN_EXPS_IN_SERIAL=False,
        INTERACT=False,
        REGEN_DATA=False,
        EPOCHS=10,
        SAVE_DATA=True,
        PIPELINE='TRAIN VAL REC',
        GET_LOGS=True,
        VERBOSE=True,
        REPEAT_ALL=1,
        NTRAIN=(25, 50, 100, 150, 200, 1000),
        BATCH_SIZE=10,
        NORM_TRAIN_IMS=True,
        OVERWRITE_NORMS=False,
        EXPS=(
                {
                    "gpus": 1,
                    "arch": "SCRATCH"
                }
                , {
                    "gpus": 1,
                    "arch": "INC"
                }
                , {
                    "gpus": 1,
                    "arch": "ALEX"
                }
                , {
                    "gpus": 1,
                    "arch": "GNET"
                }
        )
):
    return obj(locals())



DEFAULT_CONFIG = RunExpsConfig()
DEFAULT_NO_NORM = RunExpsConfig(
    NORM_TRAIN_IMS=False
)
def _FILE_OP_CONFIG(**kwargs): return RunExpsConfig(
    RUN_EXPS_IN_SERIAL=True,
    SAVE_DATA=False,
    GET_LOGS=False,
    **kwargs
)
GEN_IMAGES = _FILE_OP_CONFIG(
    EPOCHS=0,
    REGEN_DATA=True,
    EXPS=[]
)
PRE_PRE_PREPROCESS = _FILE_OP_CONFIG(
    EPOCHS=1,
    OVERWRITE_NORMS=True
)
TEST_FOUR = RunExpsConfig(
    NTRAIN=(25, 50),
    EPOCHS=2,
    EXPS=[
        {
            "gpus": 1,
            "arch": "SCRATCH"
        }
        , {
            "gpus": 1,
            "arch": "INC"
        }
    ]
)
TEST_ONE = RunExpsConfig(
    RUN_EXPS_IN_SERIAL=True,
    INTERACT=True,
    EPOCHS=2,
    NTRAIN=(25,),
    EXPS=[
        {
            "gpus": 1,
            "arch": "SCRATCH"
        }
    ]
)

EXP_MOD = selected_config()