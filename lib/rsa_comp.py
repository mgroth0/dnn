import multiprocessing
import numpy as np
from multiprocessing.managers import DictProxy
from time import sleep

from lib.datamodel.Correlational import CorrelationalArrays, Covariance, L2_Norm, PearsonCorrelation
from lib.datamodel.Mats import MathFail
from lib.rsa_comp_helpers import darius_and_shobhita_acts, RSA_LAYERS, rsa_pattern, separate_comp_mat_by_classes_compared, SYM_CLASS_SET_PATTERNS
from lib.rsa_figs import RSACorrCoefTable, RSAImagePlot, RSAPValueTable, RSAViolinBuilder
from mlib.boot import crunch
from mlib.boot.crunch import section
from mlib.boot.lang import listkeys
from mlib.boot.stream import flat, listitems
from mlib.parallel import mark_process_complete, non_daemon_process_map, print_parallel_status, run_in_thread


def main(
        N_PER_CLASS,
        ACT_SIZE,
        INCLUDE_DARIUS,
        ALL_METHODS,
        EXCLUDE_DARIUS_SMALLER_TRAIN_SIZES,
        MULTIPROCESS,
        GPU
):
    @run_in_thread
    def print_status_updates():
        while True:
            sleep(10)
            print_parallel_status()
    corr_coef_tables = {}
    def thegen():
        for arch, net, feature_mat in darius_and_shobhita_acts(
                N_PER_CLASS,
                ACT_SIZE,
                INCLUDE_DARIUS,
                EXCLUDE_DARIUS_SMALLER_TRAIN_SIZES
        ):
            yield arch, net, feature_mat, ALL_METHODS, N_PER_CLASS

    if not MULTIPROCESS:
        global_result_list = map(lambda t: process_activation_data(*t, MULTIPROCESS=MULTIPROCESS), thegen())
    else:
        print(f'cpu_count:{multiprocessing.cpu_count()}')
        with crunch.get_manager().prepare_for_multiprocessing() as manager:
            assert isinstance(crunch.get_manager().PIPELINE_SECTIONS["Startup"], DictProxy)
            global_result_list = non_daemon_process_map(
                lambda t: process_activation_data(*t, manager=manager, MULTIPROCESS=MULTIPROCESS,GPU=GPU),
                thegen(),
            )

    assert all(g is not None for g in
               global_result_list), f'BUG: why is there a None in global_result_list? global_result_list=[{",".join(str(g) for g in global_result_list)}]'

    for global_result in global_result_list:
        for method_name, data in listitems(global_result):
            coefs = data['coefs']
            net = data['net']
            layer = data['layer']
            arch = data['arch']

            TAGS = [net, layer, arch, method_name]

            if method_name in corr_coef_tables:
                corr_coef_tables[method_name].net_coefs[net] = coefs
                corr_coef_tables[method_name].tags = list(set(corr_coef_tables[method_name].tags + [net, layer, arch]))
            else:
                corr_coef_tables[method_name] = RSACorrCoefTable(
                    tags=TAGS,
                    net_coefs={net: coefs},
                    method_name=method_name,
                    EXCLUDE_DARIUS_SMALLER_TRAIN_SIZES=EXCLUDE_DARIUS_SMALLER_TRAIN_SIZES
                )

    [t.save().build() for t in corr_coef_tables.values()]


def process_activation_data(
        arch,
        net,
        feature_mat,
        ALL_METHODS,
        N_PER_CLASS,
        manager=None,
        MULTIPROCESS=False,
        GPU = False
):
    import mlib.boot.global_manager_ref
    mlib.boot.global_manager_ref.manager = manager  # NEED TO TWICE, IN THE SUBPROCESS CODE, AND IN THE PARENT CODE
    assert crunch.get_manager().PIPELINE_SECTIONS["Startup"] is not None
    def comp_mats():
        for method in [L2_Norm, PearsonCorrelation, Covariance]:
            mname = method.__name__
            with section(f"{mname}: {net}"):
                try:
                    yield method.HIGH_IS_SIMILAR, mname, feature_mat.compare(method, GPU=GPU)
                    if not ALL_METHODS:
                        break
                except MathFail:
                    print('MathFail!')
        if ALL_METHODS:
            for k in range(2, 5):
                KHIGH_IS_SIMILAR = False
                mname = f"KMeans(k={k})"
                with section(f"{mname}: {net}"):
                    yield KHIGH_IS_SIMILAR, mname, feature_mat.kcompare(k=k)


    global_results = {}

    for HIGH_IS_SIM, method_name, rsa_comp_mat in comp_mats():
        global_results[method_name] = {}
        normalized1 = rsa_comp_mat / np.nanmax(rsa_comp_mat)
        sim_string = 'Dissimilarity' if not HIGH_IS_SIM else 'Similarity'
        stat_arrays = {pat: separate_comp_mat_by_classes_compared(
            normalized1, net, arch, method_name, sim_string, pat
        ) for pat in listkeys(SYM_CLASS_SET_PATTERNS)}

        comp_mat = rsa_comp_mat.resize_if_longer_than(1000)  # large resolutions caused freezing

        layer = RSA_LAYERS[arch]

        TAGS = [net, layer, arch, method_name]

        coefs = CorrelationalArrays(
            data={
                     'activations': flat(normalized1),
                 } | {
                     pat: flat(rsa_pattern(pat, N_PER_CLASS, HIGH_IS_SIM=HIGH_IS_SIM))
                     for pat in SYM_CLASS_SET_PATTERNS.keys()
                 },
            ylabel=f'{sim_string} Score ({method_name})',
            xlabel='Full RSA output / Generated Pattern',
        ).correlate_to('activations')

        ttest_results = {pat: sas.ttests() for pat, sas in stat_arrays.items()}

        for pat, ttest_result in ttest_results.items():
            # try:
            RSAPValueTable(tags=TAGS, pat=pat, arch=arch, ttest_result=ttest_result, method_name=method_name,
                           net=net).save().build()
            # except:
            #     import mlib.err
            #     mlib.err.auto_quit_on_exception = False
            #     breakpoint()

        with section(f"building violin plots for {method_name}:{net}"):
            for pat, sas in stat_arrays.items():
                RSAViolinBuilder(
                    sas=sas,
                    tags=TAGS,
                    pat=pat,
                    sim_string=sim_string,
                    method_name=method_name,
                    arch=arch,
                    net=net
                ).save().build()

        with section(f"building image plot for {method_name}:{net}"):
            RSAImagePlot(
                tags=TAGS,
                net=net,
                layer=layer,
                comp_mat=comp_mat,
                arch=arch
            ).save().build()

        global_results[method_name] = {
            'net'  : net,
            'layer': layer,
            'arch' : arch,
            'coefs': coefs
        }
    mark_process_complete()
    return global_results
