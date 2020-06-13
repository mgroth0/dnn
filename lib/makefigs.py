import argparse
from collections import Counter

from lib.boot import bootfun  # noqa

from lib.figs.JsonSerializable import obj
from lib.defaults import *
import lib.wolf.makefigslib as makefigslib
from lib.wolf.makefigslib import getFigDats, wlexpr, wl
from lib.wolf.wolf_lang import ImageSize
# loggy.initTic()
from lib.figs import *

# from wolfpy import *


def main(overwrite=False, root=None):
    log(f'running makefigs overwrite={overwrite},root={root}')

    makefigslib.init()
    # weval('<< src/main/wolfram/util.wl')


    figDats = getFigDats(root)
    log('got ' + str(len(figDats)) + ' figDats')

    # plots_made_count = 0

    log('importing asyncio')
    import asyncio
    log('importing wolframclient')
    from wolframclient.evaluation import WolframEvaluatorPool
    log('imported wolframclient')

    async def runasync():
        log('running runasync')
        async with WolframEvaluatorPool() as pool:
            # start = time.perf_counter()
            # tasks = [
            #     pool.evaluate('Pause[1]')
            #     for i in range(10)
            # ]
            c=Counter(a=1)





            async def logAfter(wlexp,c,total):
                await pool.evaluate(wlexp)
                log(f'Finished making {c["a"]}/{total} figures')
                c['a'] += 1

            tasks = [logAfter(dealWithFD(fd, overwrite),c,len(figDats)) for fd in figDats]

            # await asyncio.wait([pool.evaluate(wlexpr('''
            # LogAfter[x] := (
            # x = Evaluate[#x]
            # Print["I did something"]
            # x
            # )
            # '''))])
            #

            log('waiting for tasks')
            await asyncio.wait(tasks)
            log('done waiting for tasks')
            # print('Done after %.02fs, using up to %i kernels.'
            #       % (time.perf_counter()-start, len(pool)))

    # python 3.5+
    log('getting asyncio event loop')
    loop = asyncio.get_event_loop()
    log('running asyncio loop for makefigs')
    loop.run_until_complete(runasync())

    # python 3.7+
    # asyncio.run(runasync())

    # for fd in figDats:
    #     dealWithFD(fd, overwrite)
    # final_t = log("Finished creating " + str(plots_made_count) + " plots!")
    log('finished making plots!')


def dealWithFD(fd, overwrite):
    log('dealing with a figDat')
    fd = obj(fd)
    log("reading:" + fd.file)
    if overwrite or not fd.pngExists and (True or not "_line" in fd.imgFile):
        if overwrite or not fd.pngExists and (True or not "_line" in fd.imgFile):
            fs = json.loads(File(fd.file).read())
            fswl = wlexpr(f'ToAssociation[Import["{fd.file}"]]')
            fs = obj(fs)
            firstLine = True
            viss = fs.viss
            setMinY = np.inf
            setMaxY = -np.inf
            makes = []
            for v in viss:
                if v.make:
                    makes.append(v)
                    if v.item_type != 'table':
                        v.minY = parse_inf(v.minY)
                        v.minX = parse_inf(v.minX)
                        v.maxX = parse_inf(v.maxX)
                        v.maxY = parse_inf(v.maxY)
                        setMinY = min(setMinY, v.minY)
                        setMaxY = min(setMaxY, v.maxY)

            if makes:
                log('showing...')
                a = []
                for idx, vis in enumerate(makes):
                    fdwl = wlexpr(f'ToAssociation[Import["{fd.file}"]][["viss",{idx + 1}]]')
                    a.append(addLayer(vis, fdwl))
                fig = wl.Show(a)

                return wl.UsingFrontEnd(wl.Export(fd.imgFile, fig, ImageSize(1000)))

                # wl.LogAfter(
                #
                # )


                # plots_made_count += 1
                # WOLFRAM.export(
                #     fd.imgFile,
                #     fig,
                #     ImageSize(1000)
                #     # ImageSize(1000, 700)
                #     # ImageSize(1000, 700)
                # )
    return '0'

def addLayer(fd, fdwl):
    DS = None  # 100 10000
    if fd.item_type != 'table':
        x = fd.x
        y = fd.y
    if fd.item_type == "table":
        return table.table(fd)
    elif fd.item_type == "line":
        return line.line(fd, fdwl, x, y, DS)
    elif fd.item_type == "scatter":
        return scatter.scatter(fd, fdwl)
    elif fd.item_type == "image":
        log('making image')
        try:
            im = wl.Image(str(fd.x).replace('[', '{').replace(']', '}'))  # "Byte"
            im = str(im).replace("'", "")
            im = weval(im)
        except:
            log('got err')
        log('made image')
        return im
    else:  # fd.item_type == "bar"
        return bar.bar(fd, x, y, fdwl)




def lastFigurableFolder(compiled=False):
    figurableFolds = File('_figs/figs_dnn').listmfiles()
    if not compiled:
        figurableFolds = listfilt(lambda f: 'compile' not in f.name, figurableFolds)
    return figurableFolds[-1].abspath if len(figurableFolds) > 0 else None