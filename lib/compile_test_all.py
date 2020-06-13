import argparse
import copy

from lib.figs.TableData import ConfusionMatrix
from lib.cta_lib import *
from lib.boot import bootfun
from lib.defaults import *
from lib.boot import loggy
from lib import makefigs

def main(root_folder, overwrite):
    log('running compile_test_all for ' + str(root_folder))
    COLORS = {
        'SCRATCH': [1, 0, 0],
        'INC'    : [1, 0, 1],
        'ALEX'   : [0, 1, 0],
        'GNET'   : [0, 0, 1]
    }
    root_folder = File(root_folder)

    metadataFile = root_folder.resolve('metadata.json')

    ARCHS = listmap(lambda a: (
        'SCRATCH' if 'SCRATCH' in a else
        'INC' if 'INC' in a else
        'ALEX' if 'ALEX' in a else
        'GNET' if 'GNET' in a else err('do not know net: ' + a))
                    , metadataFile.load()['archs'])

    NTRAINS = metadataFile.load()['ntrainims']
    NEPOCHS = metadataFile.load()['nepochs']

    compile_root = File(root_folder.parentDir).resolve(root_folder.name + '-compile')
    if overwrite:
        compile_root.deleteIfExists()
    sub_root = compile_root.resolve('1/compiled')
    sub_root.mkdirs()

    compile_root.mkdir()
    metadataFile.copy_to(compile_root)

    experiments = arr()
    for exp_folder in root_folder.listfiles():
        if File(exp_folder).name == 'metadata.json':
            continue
        exp = struct()
        exp.folder = File(exp_folder)
        exp.id = exp.folder.name
        parts = File(exp.folder.listfiles()[0]).name.split('_')
        exp.arch = parts[0]
        exp.ntrainim = parts[1]
        exp.prefix = exp.folder.abspath + '/' + exp.arch + '_' + exp.ntrainim
        experiments += exp

    TrainTable = FinalResult(2, '__test/Matthews_Correlation_Coefficient_fs.json',
                             is_table=True, rows=ARCHS, cols=NTRAINS)

    exp1 = experiments[0]
    for ex in exp1.folder.resolve(f'{exp1.arch}_{exp1.ntrainim}__train').glob('*.png'):
        classnam = ex.name_pre_ext
        ex.copy_to(
            sub_root.resolve('examples_train').resolve(f'{classnam}.png')
        )
    for ex in exp1.folder.resolve(f'{exp1.arch}_{exp1.ntrainim}__val').glob('*.png'):
        classnam = ex.name_pre_ext
        ex.copy_to(
            sub_root.resolve('examples_val').resolve(f'{classnam}.png')
        )

    for ni, ntrain in enum(NTRAINS):
        MCC = AverageResult(1, '__test/Matthews_Correlation_Coefficient_fs.json')
        for ai, arch in enumerate(ARCHS):
            results_to_compile = [
                TrainTable, MCC,
                AverageResult(2, f'__L2-Output/CM{NEPOCHS}_fs.json'),
                AverageResult(2, f'__L2-Inter/CM{NEPOCHS}_fs.json'),
                AverageResult(2, f'__L2-Raw/CM{NEPOCHS}_fs.json'),
                AverageResult(2, f'__val/CM{NEPOCHS}_fs.json', is_table=True)
            ]
            for exp in filter(lambda e:
                              e.arch == arch and e.ntrainim == str(ntrain),
                              experiments):
                for res in results_to_compile:
                    res.append(res.exp_data(exp), (ai, ni, 0), is_GNET=arch == 'GNET')
            for res in results_to_compile:
                for vis in res.j.viss: vis.make = True
                if res.dims == 1:
                    LINE_INDEX = -1
                    avg = np.mean(res.data, axis=0).tolist()
                    res.data = arr2d()
                    res.j.viss.append(copy.deepcopy(res.template))
                    res.j.viss[LINE_INDEX].make = True
                    res.j.viss[LINE_INDEX].y = avg
                    res.j.viss[LINE_INDEX].item_colors = COLORS[arch]
                    for v in itr(res.j.viss):
                        res.j.viss[v].title = f'MCC(nTrainIms={ntrain})'
                    res.j.viss[LINE_INDEX].y_label = arch
                elif res.dims == 2:
                    if isinstance(res, AverageResult):
                        avg = np.mean(res.data, axis=2)
                    else:
                        avg = res.data[:, :, 0]
                    res.j.viss[0].confuse_target = flatmax(avg)
                    # res.j.viss[0].title_size = 30
                    if res.is_table:
                        avg = np.concatenate((res.row_headers[1:], avg), axis=1)
                        avg = np.concatenate((res.col_headers, avg), axis=0)
                    if isinstance(res, AverageResult):
                        res.j.viss[0].data = avg.tolist()
                        res.j.viss[0].title = res.j.viss[0].title.replace(" ", f'{ntrain} ', 1)
                        sub_root.resolve(f'{arch}_{ntrain}{res.suffix.replace("/", "_")}').save(res.j)
                    elif ai == len(ARCHS) - 1 and ni == len(NTRAINS) - 1:
                        res.j.viss = [ConfusionMatrix(
                            data=avg.tolist(),
                            title="Final Training MCCs",
                            confuse_target=1,
                            confuse_max=1,
                            headers_included=True,
                            make=True,
                            side_header_label='Architecture',
                            top_header_label='#Train Images'
                        )]
                        sub_root.resolve(f'Final_Train_MCC_fs.json').save(res.j)

        for res in results_to_compile:
            if res.dims == 1:
                sub_root.resolve(
                    f'{res.suffix[1:].replace("_test/", "").replace("_fs", "_" + str(ntrain) + "_fs")}').save(
                    res.j)

    log('finished compilation, running makefigs')
    makefigs.dnn(root=compile_root.abspath, overwrite=overwrite)
    log('finished makefigs, back in compile_test_all')
    final_t = log('done with compile_test_all!')
    File('_logs/timelog.txt').append(f'{round(final_t)}\t\t{root_folder.name}\n')
    reloadIdeaFilesFromDisk()