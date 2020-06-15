from lib.boot import bootfun  # noqa
from lib.defaults import *
from lib.figs.JsonSerializable import obj
from lib.wolf.makefigslib import MPLFigsBackend

@log_invokation(with_args=True)
def makefigs(cfg, overwrite=False):
    root = cfg.root
    figDats = getFigDats(root)
    if cfg.fig_backend == 'wolfram':
        from lib.wolf.wolf_figs import WolfMakeFigsBackend
        backend = WolfMakeFigsBackend
    else:
        backend = MPLFigsBackend
    figDats = [obj(fd) for fd in figDats]
    figDats = [fd for fd in figDats if overwrite or not fd.pngExists]
    backend.makeAllPlots(figDats, overwrite)
    log('finished making plots!')




def lastFigurableFolder(compiled=False):
    figurableFolds = File('_figs/figs_dnn').listmfiles()
    if not compiled:
        figurableFolds = listfilt(lambda f: 'compile' not in f.name, figurableFolds)
    return figurableFolds[-1].abspath if len(figurableFolds) > 0 else None

@log_return(as_count=True)
def getFigDats(ROOT_FOLDER):
    files = []
    ids = []
    folders = []
    pngFiles = []
    fig_exps = []

    # global this_exps
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

    figDats = [{"file": File(file), "folder": folder, "id": id, "exp": exp} for file, folder, id, exp in
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

    # for key in list(possible_exps.keys()):
    #     if possible_exps[key]:
    #         this_exps = this_exps + ',' + key

    return figDats

# this_exps = ''
