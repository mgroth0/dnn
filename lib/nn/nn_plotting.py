from mlib.JsonSerializable import FigSet

from lib.dnn_data_saving import save_dnn_data
from lib.nn.nnstate import EVAL_AND_REC_EVERY_EPOCH
from mlib.fig.PlotData import PlotData
from mlib.math import safemean

def plot_metric(met, met_log, met_name):
    fs = FigSet(
        PlotData()
    )
    if EVAL_AND_REC_EVERY_EPOCH:
        fs.viss.append(PlotData())
    tit = met.title()
    fs[0].title = tit
    fs[0].title_size = 50
    x_train = []
    x_eval = []
    y_train = []
    y_eval = []
    xi = 1
    colors = []
    eval_ys = []
    for idx, the_log in enumerate(met_log):
        if 'fit' in the_log[0]:
            x_train += [int(the_log[0].split("fit")[-1])]
            xi = xi + 1
            y_train += [float(the_log[1])]
        elif EVAL_AND_REC_EVERY_EPOCH and 'eval' in the_log[0]:
            if idx == len(met_log) - 1 or 'fit' in met_log[idx + 1, 0]:
                colors += [[1, 1, 0]]
                the_x = int(the_log[0].split("eval")[-1])
                x_eval += [the_x]
                eval_x = the_x
                xi = xi + 1
                eval_ys += [float(the_log[1])]
                eval_y = safemean(eval_ys)
                y_eval += [eval_y]
                eval_ys = []

                fs.viss += [PlotData()]
                fs[-1].item_type = 'line'
                fs[-1].y = [0, 1]
                fs[-1].x = [eval_x, eval_x]
                fs[-1].item_colors = [1, 1, 1]

            else:
                eval_ys += [float(the_log[1])]

    fs[0].x = x_train
    if EVAL_AND_REC_EVERY_EPOCH:
        fs[1].x = x_eval
        fs[1].y = y_eval
        fs[1].item_type = 'scatter'
        fs[1].item_colors = colors
        fs[1].scatter_shape = '*'
        fs[0].maxX = eval_x + 1
    else:
        fs[0].maxX = max(x_train) + 1
    if 'matthew' in met:
        fs[0].minY = -1
    else:
        fs[0].minY = 0
    fs[0].maxY = 1
    fs[0].minX = 0

    fs[0].y = y_train

    fs[0].item_type = 'line'

    fs[0].item_colors = [0, 0, 1]

    save_dnn_data(fs, met_name, tit, ext='mfig')
    return fs
