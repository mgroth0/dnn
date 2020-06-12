from copy import deepcopy

from lib.defaults import *
from lib.wolf.makefigslib import DEFAULT_TICK_SIZE, OneWayOfShowingARaster, LinePlotGrid, addHeaderLabels
from lib.wolf.wolf_lang import *
# from wolfpy import weval
def table(fd):
    data = fd.data
    if fd.confuse:
        tg = fd.confuse_target
        low = 0
        high = tg
        scaleBits = []
        for i in range(1, 21):
            scaleBits += [[[0, 0, i / 21]]]
        show_nums = fd.show_nums
        backgrounds = deepcopy(data)
        for r in itr(data):
            for c in itr(data[r]):
                if data[r][c] is None:
                    data[r][c] = ''
                    backgrounds[r][c] = wlexpr('None')
                elif not isstr(data[r][c]):
                    # if r > 0 and c > 0:
                    dat = data[r][c]
                    if high != 0:
                        b = (dat / high)
                    else:
                        b = dat
                    if show_nums:
                        data[r][c] = sigfig(dat, 2)
                        # wl.Item(

                    # )



                    # Graphics(
                    # wl.Text(str(sigfig(dat, 2))),
                    # ImageSize()
                    # )
                    # Text(
                    # Item()
                    #     , Background(0, 0, b)
                    else:
                        data[r][c] = [0, 0, b]
                    if (fd.headers_included and r > 0 and c > 0) or not fd.headers_included:
                        backgrounds[r][c] = Color(0, 0, b)
        block_len = fd.block_len
        if block_len is not None and fd.headers_included is False:
            divs = [[], []]
            for i in range(len(data)):
                if i == 1 or i % block_len == 0:
                    divs[0] += [True]
                    divs[1] += [True]
                else:
                    divs[0] += [False]
                    divs[1] += [False]
    if not fd.confuse or fd.headers_included:
        for ri, row in enumerate(data):
            for ci, el in enumerate(row):
                data[ri][ci] = Item(
                    Text(el),
                    Background(backgrounds[ri][ci])
                )
        if fd.top_header_label is not None or fd.side_header_label is not None:
            data = addHeaderLabels(data, fd.top_header_label, fd.side_header_label).tolist()
    insets = [Inset(
        Rasterize(
            Grid(
                data,
                Dividers(False),
                # Background(backgrounds)
            ),
            RasterSize(),
            ImageSize(),
            Background()
        )
    )]
    # data[1][1]
    # r = wl.Rasterize(
    #     wl.Grid([[wl.Item(1)]]))
    if fd.confuse and fd.block_len is not None and fd.block_len > 1:
        if fd.confuse_is_identical:
            for r in itr(data):
                for c in itr(data[0]):
                    if c > r:
                        data[r][c] = [0, 0, 0]

        scale = Graphics(
            [
                Raster(scaleBits),
                Inset(Text(round(low), fontSize=30), [0.5, -1]),
                Inset(Text(round(high), fontSize=30), [0.5, 21]),
            ],
            ImagePadding([[75, 75], [20, 20]]),
        )

        gl = len(data)
        nt = len(fd.row_headers)
        line_values = np.linspace(fd.block_len, fd.block_len * nt, nt)
        half = fd.block_len / 2
        labellocs_labels = ziplist(line_values - half, fd.row_headers)

        # ticks start from the bottom but I want these to start from the top
        # labellocs_labels.reverse()
        gridlines = LinePlotGrid(line_values, triangle=fd.confuse_is_identical)
        # rasters start from the bottom but I want this to start from the top
        rast = OneWayOfShowingARaster(Raster(reversed(data)), gl)

        x_ticks = []
        y_ticks = []
        for t in labellocs_labels:
            x_ticks += [Text(
                t[1],
                coords=[t[0] / gl, -.02],
                direction=[1, 0],
                fontSize=DEFAULT_TICK_SIZE,
                offset=[0, 0],
            )]

        for t in labellocs_labels:
            y_ticks += [Text(
                t[1],

                # y ticks need to be reversed
                coords=[-.01, 1 - (t[0] / gl)],


                direction=[1, 0],
                fontSize=DEFAULT_TICK_SIZE,
                offset=[1, 0],
            )]

        insets = [
            Inset(
                obj=Rasterize(scale),
                pos=[1.2, 0],
                opos=[Center, Bottom]
            ),
            Inset(
                obj=Rasterize(rast,
                              ImageResolution(),
                              # RasterSize()
                              ),
                opos=[Left, Bottom]
            ),
            Inset(
                obj=Rasterize(
                    gridlines, ImageResolution(),
                    Background(
                        # wl.Red
                        wlexpr('None')
                    )
                ),
                opos=[Left, Bottom],
                background=Background(
                    # wl.Red
                    wlexpr('None')
                )
            )

        ]
        [insets.extend(ticks) for ticks in zip(x_ticks, y_ticks)]
    insets += [
        Inset(
            Rasterize(
                Text(
                    fd.title,
                    # coords=Scaled(
                    #     [0.5, .95 if fd.headers_included else 1.2]
                    # ),
                    fontSize=(40 if fd.headers_included else 20) if fd.title_size is None else fd.title_size
                ),
                Background(wlexpr('None'))
            ),
            scale=(1, 1),
            pos=Scaled([0.5, 1]),
            background=Background(wlexpr('None'))
        )
    ]
    r = Graphics(
        insets
    )

        # debug
        # r = gridlines

    return r  # weval(r)
