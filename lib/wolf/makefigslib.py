from lib.wolf.wolf_lang import *

bg = None
bgi = None


DEFAULT_TICK_SIZE = 15

from lib.defaults import *

def init():
    global bg, bgi
    bg = Black
    bgi = White



def defaultPlotOptions(fd):
    maxY = wl.All if fd.maxY is None or fd.maxY == 'inf' or not isreal(fd.maxY) else float(fd.maxY)
    minY = wl.All if fd.minY is None or fd.minY == 'inf' or not isreal(fd.minY) else float(fd.minY)
    maxX = wl.All if fd.maxX is None or fd.maxX == '-inf' or not isreal(fd.maxX) else float(fd.maxX)
    minX = wl.All if fd.minX is None or fd.minX == 'inf' or not isreal(fd.minX) else float(fd.minX)

    if maxY != wl.All and minY != wl.All:
        diff = maxY - minY
        pad = diff * (fd.y_pad_percent / 100)
        maxY = maxY + pad
        minY = minY - pad

    if maxX != wl.All and minX != wl.All:
        #     forced padding for labels
        diff = maxX - minX
        pad = diff * 0.2
        maxX = maxX + pad

    return [
        PlotLabel(Style(fd.title, FontSize(fd.title_size))),
        PlotRange([[minX, maxX], [minY, maxY]]),
        LabelStyle(bgi),
        Background(bg),
        FrameTicksStyle(Directive(bgi, 10)),
        AxesStyle(Directive(Large, bgi)),
        IntervalMarkersStyle(bgi),
        # (*    {{Large, bgi}, {Large, bgi}}*)
        Frame(True),
        FrameStyle([
            [
                #     left
                Directive(Opacity(1), FontOpacity(1), FontSize(fd.label_size)),
                #     right
                Directive(Opacity(0), FontOpacity(1), FontSize(fd.label_size))
            ],
            [
                #     bottom
                Directive(Opacity(1), FontOpacity(1), FontSize(fd.label_size)),
                #     top
                Directive(Opacity(0), FontOpacity(1), FontSize(fd.label_size))
            ]
        ]),
        LabelStyle([bgi, FontWeight("Bold"), FontSize(fd.label_size)])
    ]

# import sys
# exps = sys.argv[1].split(',')

this_exps = ''

def getFigDats(ROOT_FOLDER):
    files = []
    ids = []
    folders = []
    pngFiles = []
    fig_exps = []

    global this_exps
    possible_exps = dict()
    for name in File(ROOT_FOLDER).listfiles():
        if '.DS_Store' in name or 'metadata.json' in name: continue
        this_exp = os.path.basename(name)
        possible_exps[this_exp] = False

        for name in File(name).listfiles():
            #     if os.path.basename(name) in exps:
            for subname in File(name).listfiles():
                if subname.endswith("fs.json"):
                    files.append(subname)
                    ids.append(subname.split("/")[-1].replace("_fs.json", ""))
                    # folders.append(ROOT_FOLDER + "/" + subname.split("/")[-2])
                    folders.append(ROOT_FOLDER + "/" + this_exp + "/" + subname.split("/")[-2])
                    fig_exps.append(this_exp)

    figDats = [{"file": file, "folder": folder, "id": id, "exp": exp} for file, folder, id, exp in
               zip(files, folders, ids, fig_exps)]

    for name in File(ROOT_FOLDER).listfiles():
        if '.DS_Store' in name or 'metadata.json' in name: continue
        for name in File(name).listfiles():
            # if os.path.basename(name) in exps:
            for subname in File(name).listfiles():
                if subname.endswith(".png"):
                    pngFiles.append(subname)

    pngIDs = [x.split("/")[-1] for x in pngFiles]

    for idx, fd in enumerate(figDats):
        fd["imgFileName"] = fd["id"] + ".png"
        fd["imgFile"] = fd["folder"] + "/" + fd["imgFileName"]
        fd["pngExists"] = File(fd["imgFile"]).exists()
        if not possible_exps[fd['exp']]: possible_exps[fd['exp']] = not fd["pngExists"]

    for key in list(possible_exps.keys()):
        if possible_exps[key]:
            this_exps = this_exps + ',' + key

    return figDats








def importImage(file, caption=None):
    im = wl.Image(wl.Import(file), ImageSize(500))
    if caption:
        return wl.Labeled(im, caption, wl.Right)
    return im


def OneWayOfShowingARaster(rast, gl):
    return ListLinePlot([],
                        Prolog(rast),
                        NoTicks,
                        PlotRange([[0, gl], [0, gl]]),
                        ImageSize(1000),
                        AspectRatio()
                        )



def LinePlotGrid(ls, triangle=False):
    gl = ls[-1]
    listpoints = []

    for i in ls:
        if triangle:
            listpoints += [[[i, 0], [i, gl - i]]]
            listpoints += [[[0, i], [gl - i, i]]]
        else:
            listpoints += [[[i, 0], [i, gl]]]
            listpoints += [[[0, i], [gl, i]]]
    lines = ListLinePlot(listpoints,
                         PlotStyle([
                             [FadedOrangeMaybe, Dashed],
                             [Yellow, Dashed]
                         ]),
                         NoTicks,
                         PlotRange([[0, gl], [0, gl]]),
                         ImageSize(1000),
                         AspectRatio(),

                         background=Background(
                             # wl.Red
                             wlexpr('None')
                         )
                         )
    return lines


def addHeaderLabels(mat, top, side):
    data = objarray(mat, 2)
    # noinspection PyTypeChecker
    data = concat(
        make2d(
            [None, Item(Text(
                side,
                fontSize=30,
                direction=ROTATE_90
                # direction=[0,1]
            ))] + ((np.repeat([SpanFromAbove],
                              len(data) - 2).tolist()) if len(data) > 2 else [])
        ).T,
        data, axis=1)
    # noinspection PyTypeChecker
    data = np.insert(data, 0, [
        None, None,
        Item(Text(top, fontSize=30))
    ] + ((np.repeat([SpanFromLeft], len(data[0]) - 3).tolist()) if len(data[0]) > 2 else [])

                     , axis=0)
    return data
