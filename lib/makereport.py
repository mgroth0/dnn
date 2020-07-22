from lib.dnn_proj_struct import experiment_groups
from mlib.analyses import ANALYSES, AnalysisMode
from mlib.boot.lang import listkeys
from mlib.boot.stream import listmap, flat1, __
from mlib.file import MD_FILE, pwdf
from mlib.proj.struct import Project
from mlib.term import log_invokation
from mlib.web.database import Database
from mlib.web.makereport_lib import FigureTable, DNN_REPORT_CSS
from mlib.web.simple_admin_api import SimpleAdminAPI
from mlib.web.html import Hyperlink, Br, HTMLPage
from mlib.web.webpage import write_index_webpage

MAKEREPORT_ONLINE = True
INCLUDE_RSA_SLIDE = False

@log_invokation
def makereport():
    MR_Database = Database('makereport.json')
    MR_API = SimpleAdminAPI(MR_Database)
    toc = []
    private_toc = []
    index_root = Project.DNN_WEB_FOLDER['public']
    index_root_private = Project.DNN_WEB_FOLDER['private']
    resource_root = Project.DNN_WEB_FOLDER['resources']
    for eg in experiment_groups().filtered(
            lambda e: e.compiled
    ):
        exp_group_folder = eg.compile_folder
        FIG_FOLDER = exp_group_folder[f'1/compiled']
        md = exp_group_folder[MD_FILE].quiet()

        write_index_webpage(
            get_report(
                md,
                resources_root=FIG_FOLDER,
                exp_name=eg.name,
                index_url=f'../{index_root_private.edition_wolf_dev.name}',  # index_root_private.wcurl,
                database=MR_Database,
                api=MR_API,
                editable=True,
                web_resources_root=resource_root
            ),
            root=eg.priv_web_folder,
            resource_root_file=resource_root,
            upload_resources=False,
            WOLFRAM=MAKEREPORT_ONLINE,
            DEV=True,  # private,
        )
        private_toc.append(Hyperlink(eg.name, eg.priv_web_folder.edition_wolf_dev.wcurl))

        write_index_webpage(
            get_report(
                md,
                resources_root=FIG_FOLDER,
                exp_name=eg.name,
                index_url=f'../{index_root.edition_wolf_pub.name}',
                database=MR_Database,
                api=MR_API,
                editable=False,
                web_resources_root=resource_root,
                show=True
            ),
            root=eg.pub_web_folder,
            resource_root_file=resource_root,
            upload_resources=False,
            WOLFRAM=MAKEREPORT_ONLINE,
            DEV=False,  # public
        )
        toc.append(Hyperlink(eg.name, eg.pub_web_folder.edition_wolf_pub.wcurl))

    [toc.insert(0, Br) for _ in range(5)]
    toc.append(Hyperlink("to private version", index_root_private.edition_wolf_dev.wcurl))

    [private_toc.insert(0, Br) for _ in range(5)]
    private_toc.append(Hyperlink("to public version", index_root.edition_wolf_pub.wcurl))

    write_index_webpage(
        HTMLPage('index', *private_toc),
        root=index_root_private,
        resource_root_file=resource_root,
        upload_resources=MAKEREPORT_ONLINE,
        WOLFRAM=MAKEREPORT_ONLINE,
        DEV=True
    )
    write_index_webpage(
        HTMLPage('index', *toc),
        root=index_root,
        resource_root_file=resource_root,
        upload_resources=False,
        WOLFRAM=MAKEREPORT_ONLINE,
        DEV=False
    )


@log_invokation
def get_report(
        md,
        resources_root,
        exp_name,
        index_url,
        database: Database = None,
        api: SimpleAdminAPI = None,
        editable=False,
        web_resources_root=None,
        show=False
):
    if exp_name not in listkeys(database):
        database[exp_name] = {}

    analysis_figdata = flat1([
        a.get_report_figdata(exp_name, resources_root, database) for a in ANALYSES(mode=AnalysisMode.PIPELINE)
    ])

    all_arch_figs = []
    ntrain_figs = md.ntrainims
    nepochs = md.nepochs

    for n in ntrain_figs:
        for suffix in [
            f'__val_CM{nepochs}',
            f'__L2-Output_CM{nepochs}',
            f'__L2-Inter_CM{nepochs}',
            f'__L2-Raw_CM{nepochs}'
        ]:
            arch_figs = listmap(__.label, md.archs)
            arch_figs = listmap(lambda a: (
                resources_root[f'{a}_{n}{suffix}.png'],
                database.get_or_set_default(
                    '',
                    exp_name,
                    f'{a}_{n}{suffix}'
                )
            ), arch_figs)

            all_arch_figs.extend(arch_figs)

    mcc_name = f'Matthews_Correlation_Coefficient'

    ntrain_figs = listmap(lambda n: (
        resources_root[f'{mcc_name}_{n}.png'],
        database.get_or_set_default(
            '',
            exp_name,
            f'{mcc_name}_{n}'
        )
    ), ntrain_figs)
    doc = HTMLPage(
        f'index',
        f'Symmetry Detection Report by Matt Groth'
        , f''
        , f'{len(md.archs)} Architectures: '
        , *listmap(__.label, md.archs)
        , f'Experiments: {md.nrepeats} per architecture'
        , f'Epochs: {md.nepochs} per experiment'
        , f'Batch Size: {md.batchsize}'
        , f'Training Images: {md.ntrainims}'
        , f'Normalized Individual Images: {md.normalized}'
        , FigureTable(
            *analysis_figdata,
            (
                resources_root['Final_Train_MCC.png'],
                database.get_or_set_default(
                    '',
                    exp_name,
                    'Final_Train_MCC'
                )
            ),
            *ntrain_figs,
            (
                pwdf()["/_figs/figs_misc/RSA_patterns/RSA_patterns.001.jpeg"],
                database.get_or_set_default(
                    '',
                    exp_name,
                    'RSA_patterns'
                )
            ) if INCLUDE_RSA_SLIDE else None,
            *all_arch_figs,
            resources_root=web_resources_root,
            exp_id=exp_name,
            editable=editable
        ),
        Hyperlink('back to index', index_url),
        api.apiElements(),
        js=api.cs(),
        style=DNN_REPORT_CSS,
        show=show
    )

    return doc
