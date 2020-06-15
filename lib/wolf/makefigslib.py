from abc import ABC, abstractmethod
from copy import deepcopy

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lib.wolf.wolf_lang import *

DEFAULT_TICK_SIZE = 15
from lib.defaults import *

class MakeFigsBackend(ABC):
    DS = None  # 100 10000
    @classmethod
    def makeAllPlots(cls, figDats, overwrite):
        will_maybe_do_more = []
        for fd in figDats:
            viss = fd.file.loado().fixInfs().viss
            makes = [v for v in viss if v.make]
            if makes:
                log('showing...')
                will_maybe_do_more += [cls.export_fd(makes, fd, overwrite)]
        return will_maybe_do_more
    @classmethod
    @abstractmethod
    def export_fd(cls, makes, fd, overwrite): pass

    @classmethod
    def addLayer(cls, fd):
        try:
            return cls.__getattribute__(
                cls, fd.item_type
            ).__get__(cls, cls)(fd)
        except AttributeError:
            return cls.__getattribute__(
                MakeFigsBackend, fd.item_type
            ).__get__(cls, cls)(fd)

    @classmethod
    @abstractmethod
    def none(cls): pass

    @abstractmethod
    def color(self, *rgb): pass


    @classmethod
    @abstractmethod
    def image(cls, fd): pass

    @classmethod
    @abstractmethod
    def tableItem(cls, o, background): pass


    @classmethod
    def table(cls, fd):
        if 'Wolf' in cls.__name__:
            from lib.wolf.wolf_figs import addHeaderLabels, LinePlotGrid, WolfMakeFigsBackend, OneWayOfShowingARaster
        if cls == MPLFigsBackend:
            cls.fig = plt.figure(
                figsize=(16, 12),
                facecolor='black'
            )
            cls.ax = cls.fig.add_subplot(111, facecolor='black')
            cls.ax.axis("off")
            cls.tabl = None
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
                    backgrounds[r][c] = cls.color(0, 0, 0)
                    if data[r][c] is None:
                        data[r][c] = ''
                        if 'Wolf' in cls.__name__:
                            backgrounds[r][c] = cls.none()
                        else:
                            backgrounds[r][c] = cls.color(0, 0, 0)
                    elif not isstr(data[r][c]):
                        dat = data[r][c]
                        if high != 0:
                            b = (dat / high)
                        else:
                            b = dat
                        if show_nums:
                            data[r][c] = sigfig(dat, 2)
                        else:
                            data[r][c] = [0, 0, b]

                        if (fd.headers_included and r > 0 and c > 0) or not fd.headers_included:
                            backgrounds[r][c] = cls.color(0, 0, b)
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
                    if cls == MPLFigsBackend:
                        if cls.tabl is None:
                            cls.tabl = cls.ax.table(
                                [[1]],
                                loc='center'
                            )
                            # cls.ax.plot([1,2,3],[1,2,3])

                        # try:

                        cell = cls.tabl.add_cell(
                            ri, ci,
                            1 / len(data[0]),
                            1 / len(data),
                            loc="center",
                            facecolor=backgrounds[ri][ci],
                            text=str(el),
                            # text="TEST_TEXT",
                            # facecolor='green'
                        )
                        cell.get_text().set_fontsize(20)
                        if len(data) > 5:
                            cell.get_text().set_fontsize(10)
                        cell.get_text().set_color('white')
                        # except:
                        # cell.set_text(el)
                    else:
                        data[ri][ci] = cls.tableItem(el, backgrounds[ri][ci])
            if fd.top_header_label is not None or fd.side_header_label is not None:
                if 'Wolf' in cls.__name__:
                    data = addHeaderLabels(data, fd.top_header_label, fd.side_header_label).tolist()
                else:
                    cls.tabl.auto_set_font_size(False)
                    h = cls.tabl.get_celld()[(0, 0)].get_height()
                    w = cls.tabl.get_celld()[(0, 0)].get_width()
                    # Create an additional Header

                    # weird = "Header Header Header Header"
                    weird = fd.top_header_label * 4


                    header = [cls.tabl.add_cell(
                        -1, pos, w, h, loc="center", facecolor="red"
                    ) for pos in
                        range(1, len(data[0]) + 1)]
                    if len(header) > 2:
                        for idx, head in enum(header):
                            if idx == 0:
                                head.visible_edges = "TBL"
                            elif idx == len(header) - 1:
                                head.visible_edges = "TBR"
                            else:
                                head.visible_edges = 'TB'
                        header[1].get_text().set_text(weird)
                    elif len(header) == 2:
                        header[0].visible_edges = 'TBL'
                        header[1].visible_edges = 'TBR'
                        header[1].get_text().set_text(weird)
                    else:
                        header[0].visible_edges = 'TBLR'
                        header[0].get_text().set_text(weird)

                    # Create an additional Header
                    weird = fd.side_header_label * 4
                    header = [cls.tabl.add_cell(pos, -1, w, h, loc="center", facecolor="none") for pos in range(1, len(data) + 1)]
                    if len(header) > 2:
                        for idx, head in enum(header):
                            if idx == 0:
                                head.visible_edges = "LTR"
                            elif idx == len(header) - 1:
                                head.visible_edges = "LRB"
                            else:
                                head.visible_edges = 'LR'
                        header[1].get_text().set_text(weird)
                    elif len(header) == 2:
                        header[0].visible_edges = 'TLR'
                        header[1].visible_edges = 'BLR'
                        header[1].get_text().set_text(weird)
                    else:
                        header[0].visible_edges = 'TBLR'
                        header[0].get_text().set_text(weird)

        if cls != MPLFigsBackend:
            insets = [Inset(
                Rasterize(
                    Grid(
                        data,
                        Dividers(False),
                    ),
                    RasterSize(),
                    ImageSize(),
                    Background()
                )
            )]
        if fd.confuse and fd.block_len is not None and fd.block_len > 1:
            if fd.confuse_is_identical:
                for r in itr(data):
                    for c in itr(data[0]):
                        if c > r:
                            data[r][c] = [0, 0, 0]

            if cls != MPLFigsBackend:
                scale = Graphics(
                    [
                        Raster(scaleBits),
                        Inset(Text(round(low), fontSize=30), [0.5, -1]),
                        Inset(Text(round(high), fontSize=30), [0.5, 21]),
                    ],
                    ImagePadding([[75, 75], [20, 20]]),
                )
            else:
                # create an axes on the right side of ax. The width of cax will be 5%
                # of ax and the padding between cax and ax will be fixed at 0.05 inch.
                divider = make_axes_locatable(cls.ax)

                cax = divider.append_axes("right", size="5%", pad=0.05)

                # cmap = matplotlib.colors.Colormap('name', N=256)
                cmap = LinearSegmentedColormap.from_list(
                    'bluemap', arr(scaleBits).reshape(20, 3), N=20)
                sm = matplotlib.cm.ScalarMappable(norm=None, cmap=cmap)
                cbar = cls.fig.colorbar(
                    sm,
                    cax=cax,
                    orientation='vertical',
                    ticks=np.linspace(
                        # low,
                        0,
                        # high,
                        1,
                        num=4
                    )
                )
                cbar.ax.set_yticklabels(np.linspace(
                    low, high, num=4
                ))

            gl = len(data)
            nt = len(fd.row_headers)
            line_values = np.linspace(fd.block_len, fd.block_len * nt, nt)
            half = fd.block_len / 2
            labellocs_labels = ziplist(line_values - half, fd.row_headers)

            # ticks start from the bottom but I want these to start from the top
            # labellocs_labels.reverse()
            if 'Wolf' in cls.__name__:
                gridlines = LinePlotGrid(line_values, triangle=fd.confuse_is_identical)
            else:
                # gl = line_values[-1]
                listpoints = []
                for i in line_values:
                    if fd.confuse_is_identical:
                        listpoints += [
                            [
                                [i, gl],
                                [i, i]
                            ]
                        ]

                        listpoints += [[[i, i], [0, i]]]

                    else:
                        listpoints += [[[i, 0], [i, gl]]]
                        listpoints += [[[0, i], [gl, i]]]
                listpoints = arr(listpoints)
                for sub in listpoints:
                    cls.ax.plot(sub[:, 0], sub[:, 1], 'y--')
            # rasters start from the bottom but I want this to start from the top
            if 'Wolf' in cls.__name__:
                rast = OneWayOfShowingARaster(Raster(reversed(data)), gl)
            else:
                cls.ax.imshow(list(data))

            x_ticks = []
            xt_mpl_t = []
            xt_mpl_l = []
            y_ticks = []
            yt_mpl_t = []
            yt_mpl_l = []
            for t in labellocs_labels:
                x_ticks += [Text(
                    t[1],
                    coords=[t[0] / gl, -.02],
                    direction=[1, 0],
                    fontSize=DEFAULT_TICK_SIZE,
                    offset=[0, 0],
                )]
                xt_mpl_t += [t[0] / gl]
                xt_mpl_l += [t[1]]

            for t in labellocs_labels:
                y_ticks += [Text(
                    t[1],
                    # y ticks need to be reversed
                    coords=[-.01, 1 - (t[0] / gl)],
                    direction=[1, 0],
                    fontSize=DEFAULT_TICK_SIZE,
                    offset=[1, 0],
                )]
                # y ticks not reversed for mpl?
                yt_mpl_t += [(t[0] / gl)]
                yt_mpl_l += [t[1]]
            if 'Wolf' in cls.__name__:
                insets = [
                    Inset(
                        obj=Rasterize(scale),
                        pos=[1.2, 0],
                        opos=[Center, Bottom]
                    ),
                    Inset(
                        obj=Rasterize(rast,
                                      ImageResolution(),
                                      ),
                        opos=[Left, Bottom]
                    ),
                    Inset(
                        obj=Rasterize(
                            gridlines, ImageResolution(),
                            Background(
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
                                fontSize=(40 if fd.headers_included else 20) if fd.title_size is None else fd.title_size
                            ),
                            Background(wlexpr('None'))
                        ),
                        scale=(1, 1),
                        pos=Scaled([0.5, 1]),
                        background=Background(wlexpr('None'))
                    )
                ]
                r = Graphics(insets)
            else:
                title_obj = cls.ax.set_title(fd.title, fontSize=fd.title_size / 3)
                plt.setp(title_obj, color='w')
                c = 'w'
                cls.ax.axis(True)
                cls.ax.spines['left'].set_color(c)
                cls.ax.spines['bottom'].set_color(c)
                cls.ax.xaxis.label.set_color(c)
                cls.ax.yaxis.label.set_color(c)
                cls.ax.tick_params(axis='x', colors=c)
                cls.ax.tick_params(axis='y', colors=c)
                cls.ax.set_xticks(arr(xt_mpl_t) * gl)
                cls.ax.set_xticklabels(xt_mpl_l, rotation=90)
                # cls.ax.xticks(rotation=90)
                cls.ax.set_yticks(arr(yt_mpl_t) * gl)
                cls.ax.set_yticklabels(yt_mpl_l)

                cax.axis(True)
                cax.spines['left'].set_color(c)
                cax.spines['bottom'].set_color(c)
                cax.spines['top'].set_color(c)
                cax.spines['right'].set_color(c)
                cax.xaxis.label.set_color(c)
                cax.yaxis.label.set_color(c)
                cax.tick_params(axis='x', colors=c)
                cax.tick_params(axis='y', colors=c)
                # cax.set_xticks(arr(xt_mpl_t) * gl)
                # cax.set_xticklabels(xt_mpl_l, rotation=90)
                # cls.ax.xticks(rotation=90)
                # cax.set_yticks(arr(yt_mpl_t) * gl)
                # cax.set_yticklabels(yt_mpl_l)

        if 'Wolf' in cls.__name__:
            return r










    @classmethod
    @abstractmethod
    def line(cls, fd): pass
    @classmethod
    @abstractmethod
    def scatter(cls, fd): pass
    @classmethod
    @abstractmethod
    def bar(cls, fd): pass


# @singleton
class MPLFigsBackend(MakeFigsBackend):
    @classmethod
    def export_fd(cls, makes, fd, overwrite):
        [cls.addLayer(vis) for vis in makes]
        cls.fig.savefig(
            fd.imgFile,
            facecolor=cls.fig.get_facecolor(),
            transparent=True
        )
        plt.cla()  # Clear axis
        plt.clf()  # Clear figure
        plt.close()  # Close a figure windo

        cls.fig = None
        cls.ax = None
        cls.tabl = None

    @classmethod
    def image(cls, fd):
        plt.imshow(fd.x)
    @classmethod
    def line(cls, fd):
        maxY = None if fd.maxY is None or fd.maxY == 'inf' or not isreal(fd.maxY) else float(fd.maxY)
        minY = None if fd.minY is None or fd.minY == 'inf' or not isreal(fd.minY) else float(fd.minY)
        maxX = None if fd.maxX is None or fd.maxX == '-inf' or not isreal(fd.maxX) else float(fd.maxX)
        minX = None if fd.minX is None or fd.minX == 'inf' or not isreal(fd.minX) else float(fd.minX)

        if maxY != None and minY != None:
            diff = maxY - minY
            pad = diff * (fd.y_pad_percent / 100)
            maxY = maxY + pad
            minY = minY - pad

        if maxX != None and minX != None:
            #     forced padding for labels
            diff = maxX - minX
            pad = diff * 0.2
            maxX = maxX + pad

        # callouts (fd.callout_x) (fd.callout)

        if cls.fig is None:
            cls.fig = plt.figure(
                figsize=(16, 12),
                facecolor='black'
            )
            cls.ax = cls.fig.add_subplot(111, facecolor='black')
        cls.ax.plot(fd.x, fd.y, color=cls.color(fd.item_colors))
        title_obj = cls.ax.set_title(fd.title, fontSize=fd.title_size)
        plt.setp(title_obj, color='w')
        c = 'w'
        cls.ax.axis(True)
        cls.ax.spines['left'].set_color(c)
        cls.ax.spines['bottom'].set_color(c)
        cls.ax.xaxis.label.set_color(c)
        cls.ax.yaxis.label.set_color(c)
        cls.ax.tick_params(axis='x', colors=c)
        cls.ax.tick_params(axis='y', colors=c)
        # cls.ax.set_xticks(arr(xt_mpl_t) * gl)
        # cls.ax.set_xticklabels(xt_mpl_l, rotation=90)
        # cls.ax.xticks(rotation=90)
        # cls.ax.set_yticks(arr(yt_mpl_t) * gl)
        # cls.ax.set_yticklabels(yt_mpl_l)
        cls.ax.annotate(
            # fd.callout,
            # "some text",
            fd.y_label,
            xy=(
                fd.x[-1]
                , fd.y[-1]
            ),
            xytext=(
                (fd.x[-1] * 1.035),
                fd.y[-1]
            ),
            xycoords='data',
            horizontalalignment='left',
            verticalalignment='bottom',
            fontsize=20,
            arrowprops=dict(
                facecolor='w',
                shrink=0.05
            ),
            bbox=dict(
                boxstyle='round,pad=0.2',
                fc='yellow',
                alpha=0.3
            )
        )



    @classmethod
    def scatter(cls, fd): TODO()
    @classmethod
    def bar(cls, fd): TODO()

    @classmethod
    def none(cls): pass

    @classmethod
    def color(cls, *rgb):
        rgb = arr(rgb) * 128
        return '#%02x%02x%02x' % tuple(ints(rgb).tolist())

    @classmethod
    @abstractmethod
    def tableItem(cls, o, background):
        err('unused')
