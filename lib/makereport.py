# EXP_GROUP__FOLDER_NAME =

from mlib.JsonSerializable import obj
from lib.figapi import APIDict
from lib.makereport_lib import update_report, upload_webpage
from lib.web import HTML, Hyperlink, Br
from lib.web_widgets import FigureTable
from lib.defaults import *
TABLE_OF_CONTENTS_URL = None
DNN_PUB_REP_URL = 'https://www.wolframcloud.com/obj/9e1d2a8d-660a-4119-a31c-7ebacab6ae57'
# EXP_GROUP_NAMES = [
#     '4-AddDarkZeros-compile',
#     '5-NormSingleIms-compile'
# ]

EXP_GROUP_NAMES = listfilt(
    lambda n: 'compile' in n,
    listmap(
        lambda f: File(f).name,
        File('_figs/figs_dnn').listfiles()))

# for pruning
# EXP_GROUP_NAMES = []

MR_API = APIDict('makereport')
@log_invokation
def makereport(prune):
    contents = []
    private_contents = []
    newPubURL = 'https://www.wolframcloud.com/obj/mjgroth/dnn/index.html'
    newPrivURL = 'https://www.wolframcloud.com/obj/mjgroth/dnn_private/index.html'

    for exp in EXP_GROUP_NAMES:
        exp_name = exp.split('-')[1]
        FIG_FOLDER = File('figs_/figs_dnn/' + exp + '/1/compiled')

        md = obj(File('figs_/figs_dnn/' + exp + '/metadata.json').load())

        priv_url = upload_webpage(
            get_report(md, FIG_FOLDER, exp_name, api=MR_API, index_url=newPrivURL, exp_id=exp.split('-')[1],
                       editable=True), exp_name + '_private')

        private_contents.append(Hyperlink(exp_name, priv_url))

        url = upload_webpage(get_report(md, FIG_FOLDER, exp_name, api=MR_API, exp_id=exp.split('-')[1]), exp_name,
                             permissions="Public")

        contents.append(Hyperlink(exp_name, url))



    [contents.insert(0, Br) for _ in range(5)]
    contents.append(Hyperlink("to private version", newPrivURL))

    [private_contents.insert(0, Br) for _ in range(5)]
    private_contents.append(Hyperlink("to public version", newPubURL))

    upload_webpage(HTML(*private_contents), 'dnn_private')
    toc_url = upload_webpage(HTML(*contents), 'dnn', permissions="Public")

    update_report(toc_url,DNN_PUB_REP_URL)
    if prune:
        log('pruning unused keys')
        for k in MR_API.unusedKeys:
            log(f'pruning: {k}')
            del MR_API[k]
    refreshSafariReport()
def get_report(md, fig_folder, exp_name, api: APIDict = None, index_url=DNN_PUB_REP_URL, exp_id=None, editable=False):
    example_folds = fig_folder.glob('example*')
    all_examples = []
    for example_fold in example_folds:

        # example_fold.name + '_' +
        # api key definition is ugly. It has to be the file name since thats whats used in figure function for now

        all_examples += listmap(lambda f: (
            f.abspath,
            api[exp_name, f.name_pre_ext]
        ), example_fold.glob("*.png"))


    all_arch_figs = []
    ntrain_figs = md.ntrainims
    nepochs = md.nepochs

    log(f'getting report for {fig_folder}')

    for n in ntrain_figs:
        # f'__test_CM{nepochs}',
        for suffix in [
            f'__val_CM{nepochs}',
            f'__L2-Output_CM{nepochs}',
            f'__L2-Inter_CM{nepochs}',
            f'__L2-Raw_CM{nepochs}'
        ]:
            arch_figs = listmap(lambda a: (
                'SCRATCH' if 'SCRATCH' in a else
                'INC' if 'INC' in a else
                'ALEX' if 'ALEX' in a else
                'GNET' if 'GNET' in a else err('do not know net: ' + a))
                                , md.archs)
            arch_figs = listmap(lambda a: (
                fig_folder.respath(f'{a}_{n}{suffix}' + '.png'),
                api[exp_name, f'{a}_{n}{suffix}']
            ), arch_figs)

            all_arch_figs.extend(arch_figs)

    mcc_name = f'Matthews_Correlation_Coefficient'

    ntrain_figs = listmap(lambda n: (
        fig_folder.respath(f'{mcc_name}_{n}.png'),
        api[exp_name, f'{mcc_name}_{n}']
    ), ntrain_figs)

    doc = HTML(
        'Symmetry Detection Report by Matt Groth'
        , ''
        , str(len(md.archs)) + ' Architectures: ',
        *md.archs
        , 'Experiments: ' + str(md.nrepeats) + ' per architecture'
        , 'Epochs: ' + str(md.nepochs) + ' per experiment'
        , 'Batch Size: ' + str(md.batchsize) + ''
        , 'Training Images: ' + str(md.ntrainims)
        , 'Normalized Individual Images: ' + str(md.normalized)
        , FigureTable(
            *all_examples,
            (
                fig_folder.resolve('Final_Train_MCC.png'),
                api[exp_name, 'Final_Train_MCC']
            ),
            *ntrain_figs,
            (
                pwd()+"/_figs/figs_misc/RSA_patterns/RSA_patterns.001.jpeg",
                api[exp_name, 'RSA_patterns']
            )
            , *all_arch_figs
            , apiURL=api.apiURL, exp_id=exp_id, editable=editable),
        Hyperlink('back to index', index_url),
        *api.apiElements()
    )

    doc.js = api.js()
    return doc