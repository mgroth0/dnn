from mlib.boot import log
from mlib.boot.bootutil import pwd
from mlib.boot.mutil import add_headers_to_mat, listkeys
from mlib.fig.TableData import ConfusionMatrix
from mlib.file import File

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

root = None
def EXP_FOLDER():
    from lib.nn import nnstate
    return File(f'{pwd()}/{root}/{nnstate.FLAGS.expid}')
def savePlotAndTableData(fs, domain, nam, isFigSet=True):
    log('here1')
    fs_str = '_fs' if isFigSet else ''
    log('here2')
    ext = 'json' if isFigSet else 'png'
    log('here3')
    domain = f'__{domain}' if len(domain) > 0 else ''
    log('here4')
    from lib.nn import nnstate
    log('here5')
    filename = f'{EXP_FOLDER().abspath}/{nnstate.FLAGS.arch}_{nnstate.FLAGS.ntrain}{domain}/{nam}{fs_str}.{ext}'
    log('here6')
    file = File(filename)
    log('here7')
    file.save(fs)
    log('here8')
