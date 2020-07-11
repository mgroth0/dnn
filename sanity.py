from lib.nn.irn2 import ImageNetClasses
from lib.nn.net_mets import error_rate_core
from mlib.boot import log
from mlib.boot.mutil import maxindex, listkeys
from mlib.boot.stream import listmap, __
from mlib.file import Folder
def sanity():
    f = Folder('_data/sanity')
    # nets = [ALEX(), INC(), GNET()]
    nets = ['ALEX', 'INC', 'INC_ORIG', 'GNET', 'ALEX_MAT_TO_PY', 'GNET_MAT_TO_PY', 'INC_MAT_TO_PY']
    results = {n: {} for n in nets}
    im_f = Folder('_ImageNetTesting/unknown')
    im_f.IGNORE_DS_STORE = True
    m_classes = listmap(__.name, im_f.files)
    y_true = [int(n.split('_')[0]) for n in m_classes]
    for n in listkeys(results):
        m_data = f[n]['ImageNetActivations.mat'].load()
        d_data = f[n]['ImageNetActivations_Darius.mat'].load()
        # m_classes = m_data['filenames'] # this was before adding all labels to filesnames
        m_data = m_data['activations']
        d_data = d_data['scoreList']
        classes = ImageNetClasses
        results[n]['same_count'] = 0

        m_y_pred = []
        d_y_pred = []

        for i in range(100):
            m_choice = maxindex(m_data[i])
            m_y_pred += [m_choice]
            d_choice = maxindex(d_data[i])
            d_y_pred += [d_choice]
            if m_choice == d_choice:
                log(f'{n} classified {m_classes[i]} the same')
                log(f'\tMatt+Darius={classes[m_choice]}')
                results[n]['same_count'] += 1
            else:
                log(f'{n} did not classify {m_classes[i]} the same')
                log(f'\tMatt={classes[m_choice]}')
                log(f'\tDarius={classes[d_choice]}')
        # results[n]['correlation'] = np.corrcoef()
        # TP, FP, TN, FN, P, N = binary_results(y_true, m_y_pred)
        results[n]['matt_acc'] = 1 - error_rate_core(y_true, m_y_pred)
        # TP, FP, TN, FN, P, N = binary_results(y_true, d_y_pred)
        results[n]['darius_acc'] = 1 - error_rate_core(y_true, d_y_pred)
    for n in listkeys(results):
        log(f'{n} classified {results[n]["same_count"]}/100 images the same')
        log(f'\t{results[n]["matt_acc"]=}')
        log(f'\t{results[n]["darius_acc"]=}')
