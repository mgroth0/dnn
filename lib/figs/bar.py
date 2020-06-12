from wolframclient.language import wl, wlexpr

import lib.wolf.makefigslib as makefigslib
from lib.defaults import *
# from wolfpy import weval
def bar(fd,x,y,fdwl):
    maxY = wl.All if fd.maxY is None or fd.maxY == -np.inf else fd.maxY
    minY = wl.All if fd.minY is None or fd.maxY == np.inf else fd.minY
    maxX = wl.All if fd.maxX is None or fd.maxY == -np.inf else fd.maxX
    minX = wl.All if fd.minX is None or fd.maxY == np.inf else fd.minX

    if maxY != wl.All and minY != wl.All:
        diff = maxY - minY
        pad = diff * 100 / fd.y_pad_percent
        maxY = maxY + pad
        minY = minY - pad

    rotate = sess.eval('90 Degree') if fd.bar_sideways_labels else sess.eval('0 Degree')

    err = fd.err
    rotate = '90 Degree' if fd.bar_sideways_labels else '0 Degree'
    vstring = '\n\n\n' if fd.bar_sideways_labels else ''
    labs = [wl.Text(xx) for xx in x]
    map_expr = wlexpr('Around[#1, #2] &')

    labs = str(fd.x).replace('[', '{').replace(']', '}').replace("'", '"')

    firstarg = wl.MapThread(map_expr, [y, err])
    origarg = firstarg
    if fd.delta_bar_idx is not None:
        # thing[]
        firstarg = weval(firstarg)
        firstarg = list(firstarg)
        firstarg[fd.delta_bar_idx] = wl.Callout(firstarg[fd.delta_bar_idx], "p=" + str(fd.delta_val), wl.Above,
                                                wl.Rule(wl.LabelStyle, [10, wl.Bold, wl.White]),
                                                wl.Rule(wl.LeaderSize, 25))

    return weval(
        wl.BarChart(firstarg,
                    makefigslib.defaultPlotOptions(fd),
                    wl.Rule(wl.ChartStyle, [wl.RGBColor(color) for color in fd.item_colors]),
                    wl.Rule(wl.LabelingFunction,
                            wlexpr('(Placed[Rotate[' + labs + '[[#2[[2]]]], ' + rotate + '], Below]&)')),
                    wl.Rule(wl.FrameTicks, [[True, wlexpr('None')], [wlexpr('None'), wlexpr('None')]]),
                    wl.Rule(wl.FrameLabel, [vstring + fd.x_label, fd.y_label])
                    )
    )