import copy

import numpy as np

from lib.cta_lib import FinalResult, AverageResult
from lib.dnn_proj_struct import experiments_from_folder, DNN_ExperimentGroup
from mlib.analyses import ANALYSES, AnalysisMode
from mlib.boot import log
from mlib.boot.lang import enum
from mlib.boot.stream import listmap, __, arr2d, itr, flatmax
from mlib.fig.PlotData import CONTRAST_COLORS
from mlib.fig.TableData import ConfusionMatrix
from mlib.str import StringExtension
from mlib.term import log_invokation


@log_invokation(with_args=True)
def analyze_exp_group(
        eg: DNN_ExperimentGroup
):
    eg.compile_folder.deleteIfExists()
    eg.metadata.copy_into(eg.compile_folder)

    ARCH_LABELS = listmap(
        __['label'],
        eg.metadata['archs']
    )
    NTRAINS = eg.metadata['ntrainims']
    NEPOCHS = eg.metadata['nepochs']

    [a.during_compile(eg) for a in ANALYSES(mode=AnalysisMode.PIPELINE)]

    experiments = experiments_from_folder(eg.folder)
    random_exp = experiments[0]

    TrainTable = FinalResult(2, 'test/Matthews_Correlation_Coefficient.mfig',
                             data_exists=random_exp.folder[f'test'].exists,
                             is_table=True, rows=ARCH_LABELS, cols=NTRAINS)

    def maybe_avg_result(pre, nepochs, is_table=False, dims=2, suf=None):
        return AverageResult(
            dims,
            f'{pre}/CM{nepochs}.mfig' if suf is None else f'{pre}/{suf}',
            data_exists=random_exp.folder[pre].exists,
            is_table=is_table
        )

    results_to_compile = []
    for ni, ntrain in enum(NTRAINS):
        MCC = maybe_avg_result(
            'test',
            None,
            dims=1,
            suf='Matthews_Correlation_Coefficient.mfig'
        )
        for ai, arch in enum(ARCH_LABELS):
            results_to_compile = [
                TrainTable, MCC,
                maybe_avg_result(f'L2-Output', NEPOCHS),
                maybe_avg_result(f'L2-Inter', NEPOCHS),
                maybe_avg_result(f'L2-Raw', NEPOCHS),
                maybe_avg_result(f'val', NEPOCHS, is_table=True)
            ]
            results_to_compile = [r for r in results_to_compile if r is not None]
            for exp in experiments.filtered(
                    lambda e: e.arch == arch and e.ntrain == ntrain,
            ):
                for res in results_to_compile:
                    res.append(res.exp_data(exp), (ai, ni, 0), is_GNET=arch == 'GNET')

            for res in results_to_compile:
                if not res.data_exists: continue
                if res.j is None:
                    log('about to breakpoint')
                    breakpoint()
                else:
                    log('res.j is not none, so no breakpoint')
                for vis in res.j.viss: vis.make = True
                if res.dims == 1:
                    LINE_INDEX = -1
                    avg = np.mean(res.data, axis=0).tolist()
                    res.data = arr2d()
                    res.j.viss.append(copy.deepcopy(res.template))
                    res.j.viss[LINE_INDEX].make = True
                    res.j.viss[LINE_INDEX].y = avg
                    res.j.viss[LINE_INDEX].item_colors = CONTRAST_COLORS[ai]
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
                        eg.compile_exp_res_folder[f'{arch}_{ntrain}__{res.suffix.replace("/", "_")}'].save(res.j)
                    elif ai == len(ARCH_LABELS) - 1 and ni == len(NTRAINS) - 1:
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
                        eg.compile_exp_res_folder[f'Final_Train_MCC.mfig'] = res.j

        for res in [mcc for mcc in results_to_compile if mcc.dims == 1 and mcc.data_exists]:
            eg.compile_exp_res_folder[
                StringExtension(res.suffix[1:]).r({
                    "test/": "",
                    "."    : f"_{ntrain}."
                })
            ].save(res.j)
