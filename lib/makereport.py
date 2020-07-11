
from mlib.boot.bootutil import pwd
from mlib.boot.mutil import err, listkeys
from mlib.boot.stream import listmap
from mlib.proj.struct import Project
from mlib.term import log_invokation
from mlib.web.database import Database
from mlib.web.makereport_lib import write_webpage, FigureTable, DNN_REPORT_CSS
from mlib.web.simple_admin_api import SimpleAdminAPI
from mlib.web.web import Hyperlink, Br, HTMLPage

@log_invokation
def makereport():
    MR_Database = Database('makereport.json')
    MR_API = SimpleAdminAPI(MR_Database)
    toc = []
    private_toc = []
    index_root = Project.DNN_FIGS_FIGS_FOLDER['0-web'].mkdir()
    index_root_private = Project.DNN_FIGS_FIGS_FOLDER['0-web-private'].mkdir()
    resource_root = Project.DNN_FIGS_FIGS_FOLDER['0-web-resources'].mkdir()
    for exp in Project.DNN_FIGS_FIGS_FOLDER.files.arrayof.name.filt_includes(
            'compile'
    ):
        exp_name = exp.split('-')[1].replace('-compile', '')
        exp_group_folder = Project.DNN_FIGS_FIGS_FOLDER[exp]
        FIG_FOLDER = exp_group_folder[f'1/compiled']
        md = exp_group_folder[f'metadata.json'].quiet()

        exp_root_priv = exp_group_folder.parent[exp.replace('-compile', '-web-private')]
        write_webpage(
            get_report(
                md,
                resources_root=FIG_FOLDER,
                exp_name=exp_name,
                index_url=f'../{index_root_private.edition_wolf_dev.name}',  # index_root_private.wcurl,
                database=MR_Database,
                api=MR_API,
                exp_id=exp.split('-')[1],
                editable=True,
                web_resources_root=resource_root
            ),
            root=exp_root_priv,
            resource_root_file=resource_root,
            upload_resources=False,
            WOLFRAM=True,
            DEV=True,  # private,
        )
        private_toc.append(Hyperlink(exp_name, exp_root_priv.edition_wolf_dev.wcurl))

        exp_root = exp_group_folder.parent[exp.replace('-compile', '-web')]
        write_webpage(
            get_report(
                md,
                resources_root=FIG_FOLDER,
                exp_name=exp_name,
                index_url=f'../{index_root.edition_wolf_pub.name}',
                database=MR_Database,
                api=MR_API,
                exp_id=exp.split('-')[1],
                editable=False,
                web_resources_root=resource_root
            ),
            root=exp_root,
            resource_root_file=resource_root,
            upload_resources=False,
            WOLFRAM=True,
            DEV=False,  # public
        )
        toc.append(Hyperlink(exp_name, exp_root.edition_wolf_pub.wcurl))

    [toc.insert(0, Br) for _ in range(5)]
    toc.append(Hyperlink("to private version", index_root_private.edition_wolf_dev.wcurl))

    [private_toc.insert(0, Br) for _ in range(5)]
    private_toc.append(Hyperlink("to public version", index_root.edition_wolf_pub.wcurl))

    write_webpage(
        HTMLPage('index', *private_toc),
        root=index_root_private,
        resource_root_file=resource_root,
        upload_resources=True,
        WOLFRAM=True,
        DEV=True
    )
    write_webpage(
        HTMLPage('index', *toc),
        root=index_root,
        resource_root_file=resource_root,
        upload_resources=False,
        WOLFRAM=True,
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
        exp_id=None,
        editable=False,
        web_resources_root=None
):
    if exp_name not in listkeys(database):
        database[exp_name] = {}

    all_examples = []
    for example_fold in resources_root.glob('example*'):
        for example_im in example_fold.glob("*.png"):
            example_type = example_fold.name.replace('examples_', '')
            if example_im.name.startswith(example_type): continue
            single_file = example_im.copy_to(example_im.parent[f'{example_type}-{example_im.name}'])
            all_examples += [(
                single_file.abspath,
                database.get_or_set_default(
                    '',
                    exp_name,
                    single_file.name_pre_ext
                )
            )]

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
            arch_figs = listmap(lambda a: (
                'SCRATCH' if 'SCRATCH' in a else
                'INC' if 'INC' in a else
                'ALEX' if 'ALEX' in a else
                'GNET' if 'GNET' in a else err(f'do not know net: {a}'))
                                , md.archs)
            arch_figs = listmap(lambda a: (
                resources_root.respath(f'{a}_{n}{suffix}.png'),
                database.get_or_set_default(
                    '',
                    exp_name,
                    f'{a}_{n}{suffix}'
                )
            ), arch_figs)

            all_arch_figs.extend(arch_figs)

    mcc_name = f'Matthews_Correlation_Coefficient'

    ntrain_figs = listmap(lambda n: (
        resources_root.respath(f'{mcc_name}_{n}.png'),
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
        , f'{len(md.archs)} Architectures: ',
        *md.archs
        , f'Experiments: {md.nrepeats} per architecture'
        , f'Epochs: {md.nepochs} per experiment'
        , f'Batch Size: {md.batchsize}'
        , f'Training Images: {md.ntrainims}'
        , f'Normalized Individual Images: {md.normalized}'
        , FigureTable(
            *all_examples,
            (
                resources_root.resolve('Final_Train_MCC.png'),
                database.get_or_set_default(
                    '',
                    exp_name,
                    'Final_Train_MCC'
                )
            ),
            *ntrain_figs,
            (
                f"{pwd()}/_figs/figs_misc/RSA_patterns/RSA_patterns.001.jpeg",
                database.get_or_set_default(
                    '',
                    exp_name,
                    'RSA_patterns'
                )
            ),
            *all_arch_figs,
            resources_root=web_resources_root,
            exp_id=exp_id,
            editable=editable
        ),
        Hyperlink('back to index', index_url),
        *api.apiElements(),
        js=api.cs(),
        style=DNN_REPORT_CSS
    )

    return doc
