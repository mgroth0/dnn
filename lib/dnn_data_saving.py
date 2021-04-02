from mlib.boot.lang import listkeys
from mlib.fig.TableData import ConfusionMatrix
from mlib.fig.figutil import add_headers_to_mat
from mlib.file import pwdf

def saveTestValResults(ARCH, nam, ds, ei):
    from lib.nn import net_mets
    from lib.nn import nnstate
    save_dnn_data(ConfusionMatrix(
        data=add_headers_to_mat(
            net_mets.cmat,
            sorted(listkeys(nnstate.CURRENT_PRED_MAP), key=lambda x: nnstate.CURRENT_PRED_MAP[x]),
            sorted(listkeys(nnstate.CURRENT_TRUE_MAP), key=lambda x: nnstate.CURRENT_TRUE_MAP[x]),
            alphabetize=True
        ).T,
        title=f'{ARCH} E{ei + 1}',
        confuse_max=len(ds) / nnstate.num_pred_classes(),
        headers_included=True
    ), f'{nam}', f'CM{ei + 1}', 'mfig')

root = None
def EXP_FOLDER():
    from lib.nn import nnstate
    return pwdf()[root][nnstate.FLAGS.expid]

def save_dnn_data(data, domain, nam, ext):
    assert len(domain) > 0
    from lib.nn import nnstate
    EXP_FOLDER().md_file['arch'] = nnstate.FLAGS.arch
    EXP_FOLDER().md_file['ntrainim'] = nnstate.FLAGS.ntrain
    EXP_FOLDER()[domain][f'{nam}.{ext}'].save(data)
