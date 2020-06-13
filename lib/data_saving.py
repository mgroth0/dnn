from lib.figs.TableData import ConfusionMatrix
from lib.defaults import *

def saveTestValResults(ARCH, nam, ds, ei):
    from lib.nn import net_mets
    from lib.nn import nnstate
    savePlotAndTableData(ConfusionMatrix(
        data=add_headers_to_mat(
            net_mets.cmat,
            listkeys(nnstate.CURRENT_PRED_MAP),
            listkeys(nnstate.CURRENT_TRUE_MAP),
            alphabetize=True
        ).T,
        title=f'{ARCH} E{ei + 1}',
        confuse_max=len(ds),
        confuse_target=len(ds) / nnstate.num_pred_classes(),
        headers_included=True
    ), f'{nam}', f'CM{ei + 1}')

def EXP_FOLDER(root):
    from lib.nn import nnstate
    return File(f'{pwd()}/{root}/{nnstate.FLAGS.expid}')
root = None
def savePlotAndTableData(fs, domain, nam,isFigSet=True):
    fs_str = '_fs' if isFigSet else ''
    ext = 'json' if isFigSet else 'png'
    domain = f'__{domain}' if len(domain) > 0 else ''
    from lib.nn import nnstate
    File(f'{EXP_FOLDER(root).abspath}/{nnstate.FLAGS.arch}_{nnstate.FLAGS.ntrain}{domain}/{nam}{fs_str}.{ext}').save(fs)
