from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto

import json
import numpy as np
import scipy
import tensorflow as tf

from arch.model_wrapper import ModelWrapper, chain_predict, simple_predict
from lib.dnn_analyses import PostBuildAnalysis
from lib.dnn_data_saving import save_dnn_data
from lib.dnn_proj_struct import DNN_ExperimentGroup, experiments_from_folder
from lib.nn.net_mets import error_rate_core
from lib.preprocessor import preprocessors
from mlib.analyses import cell, CellInput, shadow, ShadowFigType
from mlib.boot import log
from mlib.boot.lang import enum, isstr, listkeys, isint
from mlib.boot.stream import listitems, arr, listmap, __, concat, make3d, zeros, maxindex, ints, isnan, nans
from mlib.fig.text_table_wrap import TextTableWrapper
from mlib.file import File, Folder
from mlib.web.html import H3, HTML_Pre, Div, Table, TableRow, DataCell
class SanitySet(Enum):
    Set100 = auto()
    # Set50000 = auto()
@dataclass
class RealImageNetSet:
    num: int
    def __post_init__(self):
        assert 49999 >= self.num > 1

SANITY_SET = RealImageNetSet(200)  # 49999


class SanityAnalysis(PostBuildAnalysis):
    SHOW_SHADOW = True

    @cell()
    def temp_map_filenames(self):
        indexs = []
        log('loading ims...')
        old_ims = [f.load() for f in Folder('_ImageNetTesting_old')]
        new_ims = [f.load() for f in Folder('_ImageNetTesting/unknown')]
        for oi, new_im in enum(new_ims):
            log(f'checking new im {oi}...')
            for i, old_im in enum(old_ims):
                if np.all(old_im == new_im):
                    log(f'\tfound! @ {i}')
                    indexs += [i]
                    break
            assert len(indexs) == oi + 1
        File('image_net_map.p').save(indexs)
        return None



    def after_build(self, FLAGS, tf_net: ModelWrapper):
        if tf_net.pretrained and 'SANITY' in FLAGS.pipeline:
            IN_files = tf_net.IMAGE_NET_FOLD['unknown'].files

            r = {
                'files': IN_files.map(__.name),
                'ml'   : {},
                'tf'   : {}
                # 'ml2tf': {}
            }

            # ml2tf_net = tf_net.from_ML_vers().build()

            for pp_name, pp in listitems(preprocessors(tf_net.hw)):
                # , r['ml2tf'][pp_name] =
                if SANITY_SET != SanitySet.Set100:
                    root = Folder('/matt/data/ImageNet/output')
                    filenames = root.glob('validation*').map(lambda f: f.abspath).tolist()
                    def input_files():
                        ds = tf.data.TFRecordDataset(filenames)

                        image_feature_description = {
                            'image/height'      : tf.io.FixedLenFeature([], tf.int64),
                            'image/width'       : tf.io.FixedLenFeature([], tf.int64),
                            'image/colorspace'  : tf.io.FixedLenFeature([], tf.string),
                            'image/channels'    : tf.io.FixedLenFeature([], tf.int64),
                            'image/class/label' : tf.io.FixedLenFeature([], tf.int64),
                            'image/class/synset': tf.io.FixedLenFeature([], tf.string),
                            'image/class/text'  : tf.io.FixedLenFeature([], tf.string),
                            # 'image/object/bbox/xmin' : tf.io.FixedLenFeature([], tf.float32),
                            # 'image/object/bbox/xmax' : tf.io.FixedLenFeature([], tf.float32),
                            # 'image/object/bbox/ymin' : tf.io.FixedLenFeature([], tf.float32),
                            # 'image/object/bbox/ymax' : tf.io.FixedLenFeature([], tf.float32),
                            # 'image/object/bbox/label': tf.io.FixedLenFeature([], tf.int64),
                            'image/format'      : tf.io.FixedLenFeature([], tf.string),
                            'image/filename'    : tf.io.FixedLenFeature([], tf.string),
                            'image/encoded'     : tf.io.FixedLenFeature([], tf.string),
                        }
                        for raw_record in ds:
                            example = tf.io.parse_single_example(raw_record, image_feature_description)
                            yield example
                    y_true = []
                    ifs_for_labels = input_files()
                    for i in range(SANITY_SET.num):
                        y_true.append(next(ifs_for_labels)['image/class/label'].numpy())
                    r[f'tf']['y_true'] = y_true
                    def input_file_raws():
                        gen = input_files()
                        for example in gen:
                            yield tf.image.decode_jpeg(example['image/encoded'], channels=3).numpy()
                    IN_files = input_file_raws()

                # ALL = 49999
                # TEST = 10
                r[f'tf'][pp_name] = simple_predict(
                    tf_net,  # ,ml2tf_net
                    pp,
                    IN_files,
                    length=SANITY_SET.num,
                    # length=50000
                )
                # else:
                #     y_pred = V_Stacker()
                #     # root = Folder('/xboix/data/ImageNet/raw-data/validation')
                #     root = Folder('/matt/data/ImageNet/output')
                #     filenames = root.glob('validation*').tolist()
                #     ds = tf.data.TFRecordDataset(filenames)
                # #     for subroot in root:
                #         for imgfile in subroot:
                #             y_pred += tf_net.net.predict(dset, verbose=1)
                #     r[f'tf'][pp_name] = y_pred
                #     if tf_net.OUTPUT_IDX is not None:
                #         r[f'tf'][pp_name] = r[f'tf'][pp_name][tf_net.OUTPUT_IDX]

            for pp_name in ['none', 'divstd_demean', 'unit_scale', 'demean_imagenet', 'DIIL']:
                r['ml'][pp_name] = Folder('_data/sanity')[tf_net.label][
                    f'ImageNetActivations_Darius_{pp_name}.mat'
                ].load()['scoreList']

                # this was for before when darius was using the old order of activations
                # [
                #                    File('image_net_map.p').load(), :
                #                    ]

            save_dnn_data(
                data=r,
                domain='sanity',
                nam='sanity',
                ext='pickle'
            )


    def during_compile(self, eg: DNN_ExperimentGroup):
        log('about to compile_eg')
        data = self.compile_eg(eg)
        log('about to calc_accs')
        accs = self.calc_accs(data)
        if SANITY_SET == SanitySet.Set100:
            log('about to same_count_cmat')
            data = self.same_count_cmat(accs)
        log('about to acc_table')
        div = self.acc_table(accs)
        from mlib.proj.struct import Project
        f = Project.DOCS_FOLDER.edition_local['results.html']
        log(f'writing div to {f}')
        f.write(div.getCode(None, None))
        log(f'wrote div! ({f.abspath})')
        txt = f.read()
        # txt = '\n'.join(txt.split('\n')[:])
        log(f'PROOF: {f.read()[1:50]}')  # not doing whole thing bc im getting a weight EOF error

        # self.save(data)



    @shadow(ftype=ShadowFigType.NONE)
    @cell(inputs=CellInput.CACHE)
    def compile_eg(self, eg: DNN_ExperimentGroup):
        experiments = experiments_from_folder(eg.folder)
        random_exp = experiments[0]

        finished_archs = []

        pname = 'sanity.pickle'

        data = {
            k: {} for k in listkeys(random_exp.folder[f'sanity/{pname}'].load())
        }

        data['dest'] = eg.compile_exp_res_folder[pname].abspath

        for exp in eg.experiments:
            if exp.arch in finished_archs: continue
            mat = exp.folder['sanity'][pname].load()

            for backendkey, bedata in listitems(mat):
                data[backendkey][exp.arch] = bedata
                if 'y_true' in bedata:
                    data[backendkey]['y_true'] = bedata['y_true']


            finished_archs += [exp.arch]
        data['files'] = data['files'][exp.arch]
        return data

    @shadow(ftype=ShadowFigType.PREVIEW)
    @cell(inputs=compile_eg)
    def calc_accs(self, data):
        # breakpoint()
        y_true = [int(n.split('_')[0]) for n in data['files']]
        data['ml']['y_true'] = y_true
        if SANITY_SET == SanitySet.Set100:
            data['tf']['y_true'] = y_true
        # else:
        #     y_true = []
        #     for i in range(1000):
        #         y_true.extend([i] * 50)
        #     y_true = y_true[0:SANITY_SET.num]
        #     data['tf']['y_true'] = y_true
        for bekey, bedata in listitems(data):
            if bekey in ['files', 'dest']: continue  # , 'y_true'
            for akey, arch_data in listitems(bedata):
                if akey in ['y_true']: continue  # , 'y_true'
                for ppkey, ppdata in listitems(arch_data):
                    if ppkey in ['y_true']: continue
                    y_true = bedata['y_true']
                    y_pred = [maxindex(ppdata[i]) for i in range(len(ppdata))]
                    acc = 1 - error_rate_core(y_true, y_pred)
                    top5_score = 0
                    for i in range(len(y_pred)):
                        preds = maxindex(ppdata[i], num=5)
                        if y_true[i] in preds:
                            top5_score += 1
                    acc5 = top5_score / len(y_pred)
                    pp = {
                        'acts'  : ppdata,
                        'y_pred': y_pred,
                        'acc'   : acc,
                        'acc5'  : acc5
                    }
                    arch_data[ppkey] = pp
        return data

    ppdict = {
        'none'                : '0',
        'divstd_demean'       : '1',
        'unit_scale'          : '2',
        'demean_imagenet'     : '3',
        'zerocenter'          : '4',
        'demean_imagenet_crop': '5',
        'DIIL'                : '6'
    }
    bedict = {
        'tf'   : 'T',
        'ml'   : 'M',
        'ml2tf': '>'
    }
    adict = {
        'ALEX': 'A',
        'GNET': 'G',
        'INC' : 'I',
    }
    @shadow(ftype=ShadowFigType.RAW)
    @cell(inputs=calc_accs)
    def acc_table(self, data):
        titles = {
            'tf': f'Tensorflow ({100 if SANITY_SET == SanitySet.Set100 else SANITY_SET.num})',
            # 'ml2tf': 'MATLAB model imported into Tensorflow',
            'ml': 'MATLAB (100)'
        }
        sanity_report_figdata = []
        for be_key in listkeys(titles):
            be_data = data[be_key]
            if be_key in ['files', 'dest']: continue  # , 'y_true'
            arch_rows = []
            for akey, adata in listitems(be_data):
                if akey in ['y_true']: continue  # , 'y_true'
                top_row = ['Arch']
                ar = [akey]
                for ppkey, ppdata in listitems(adata):
                    top_row += [ppkey]
                    ar += [str(int(ppdata['acc'] * 100)) + '\n' + str(int(ppdata['acc5'] * 100))]
                arch_rows += [ar]
            table = [top_row] + arch_rows
            sanity_report_figdata += [H3(titles[be_key])]
            sanity_report_figdata += [HTML_Pre(str(TextTableWrapper(
                data=table,
                col_align='c' * len(table[0]),
                col_valign='m' * len(table[0])
            )))]
            if be_key == 'ml2tf':
                sanity_report_figdata += ['* Darius has uploaded new models that have not yet been tested']
        sanity_report_figdata += [H3('ImageNet Results from Literature')]
        sanity_report_figdata += [HTML_Pre(str(TextTableWrapper(
            data=[
                ['Arch', 'lit'],
                ['ALEX', f'?\n{int(0.847 * 100)}'],
                ['GNET', f'?\n{int(0.99333 * 100)}'],
                ['INC', f'80.4\n95.3']
            ],
            col_align='c' * 2,
            col_valign='m' * 2
        )))]
        sanity_report_figdata += [HTML_Pre('''
            Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. 
            "Imagenet classification with deep convolutional neural networks." 
            Advances in neural information processing systems. 2012.

            
            Szegedy, Christian, et al. 
            "Going deeper with convolutions." 
            Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
            
            Improving Inception and Image Classification in TensorFlow.” 
            Google AI Blog, 31 Aug. 2016,
             ai.googleblog.com/2016/08/improving-inception-and-image.html.
            ''')]
        return Div(*sanity_report_figdata)


    def iconfuse(self, li, lamb, identiy=True):
        cmat = nans(len(li))
        for ri, r1 in enum(li):
            for ci, r2 in enum(li):
                if not identiy or (ri >= ci):
                    same_count = lamb(r1, r2)
                    cmat[ri, ci] = same_count
                else:
                    cmat[ri, ci] = 0
        return cmat

    def confuse_analysis(self, data, lamb, identiy=True):
        @dataclass
        class IN_Result:
            backend: str
            arch: str
            pp: str
            y_pred: np.ndarray
            acts: np.ndarray
            def __str__(self):
                be = SanityAnalysis.bedict[self.backend]
                a = SanityAnalysis.adict[self.arch]
                p = SanityAnalysis.ppdict[self.pp]
                return f'{be}{a}{p}'
        in_results = []
        for bekey, bedata in listitems(data):
            if bekey in ['files', 'dest', 'y_true']: continue
            for akey, arch_data in listitems(bedata):
                for ppkey, ppdata in listitems(arch_data):
                    in_results += [IN_Result(
                        backend=bekey,
                        arch=akey,
                        pp=ppkey,
                        y_pred=arr(ppdata['y_pred']),
                        acts=arr(ppdata['acts'])
                    )]

        cmat = self.iconfuse(
            in_results,
            lamb,
            identiy=identiy
        )

        labels = listmap(
            lambda r: str(r),
            in_results
        )
        top = [None] + labels
        cmat = cmat.tolist()
        for i, li in enum(cmat):
            cmat[i] = [labels[i]] + cmat[i]
        cmat = [top] + cmat
        return cmat

    @cell(inputs=calc_accs)
    def same_count_cmat(self, data):
        return self.confuse_analysis(
            data,
            lambda r1, r2: sum(r1.y_pred == r2.y_pred)
        )

    @staticmethod
    def disp_dicts(*d):
        s = ''
        for dd in d:
            s += json.dumps(dd, indent=4) + '\n\n'
        return s



    def cmat_texttable(self, cmat, vectorfun=None):
        data = cmat if vectorfun is None else vectorfun(cmat)

        return Div(

            HTML_Pre(

                self.disp_dicts(self.bedict, self.adict, self.ppdict) + str(TextTableWrapper(
                    data=data,
                    max_width=200,
                    col_align='c' * len(cmat),
                    col_valign='m' * len(cmat)
                ))
            )
        )

    def cmat_html_table(self, cmat, vectorfun=None, int_thresh=None):
        data = cmat if vectorfun is None else vectorfun(cmat)

        div = Div()

        if int_thresh is not None:
            for ri, r in enum(data):
                for ci, e in enum(r):
                    style = {
                        'text-align': 'center'
                    }
                    if isint(e) and e >= int_thresh:
                        style.update({
                            'color': 'blue'
                        })

                    data[ri, ci] = HTML_Pre(str(e), style=style).getCode(None, None)

        div += HTML_Pre(self.disp_dicts(self.bedict, self.adict, self.ppdict))

        div += Table(
            *[TableRow(
                *[DataCell(
                    str(e),
                    style={'border': '1px solid white'} if ri > 0 and ci > 0 else {}
                ) for ci, e in enum(row)],
                style={'border': '1px solid white'}
            ) for ri, row in enum(data)],
            style={'border': '1px solid white'}
        )

        return div

    @shadow(ftype=ShadowFigType.RAW)
    @cell(inputs=same_count_cmat)
    def same_count_texttable(self, cmat):
        return self.cmat_texttable(cmat)

    @shadow(ftype=ShadowFigType.IM)
    @cell(inputs=same_count_cmat)
    def same_count_im(self, cmat):
        @np.vectorize
        def pixel_scale(e, minn, maxx):
            return 256 * ((e - minn) / maxx - minn)
        cmat = pixel_scale(arr(cmat)[1:, 1:], 0, len(cmat) - 1)
        cmat.shape = tuple(list(cmat.shape) + [1])
        cmat = concat(make3d(zeros(len(cmat))), make3d(zeros(len(cmat))), cmat, axis=2)
        return cmat


    @cell(inputs=calc_accs)
    def cross_correlate_cmat(self, data):
        def correlate_sum(r1, r2):
            r = 0
            for i in range(len(r1.acts)):
                r += np.correlate(r1.acts[i], r2.acts[i])[0]
            return r
        return self.confuse_analysis(
            data,
            correlate_sum
        )

    @shadow(ftype=ShadowFigType.RAW)
    @cell(inputs=cross_correlate_cmat)
    def cross_correlate_texttable(self, cmat):
        return self.cmat_texttable(cmat)

    @cell(inputs=calc_accs)
    def spearman_r_cmat(self, data):
        def spearman_r(r1, r2):
            r = 0
            for i in range(len(r1.acts)):
                r1copy = deepcopy(r1.acts[i])
                r2copy = deepcopy(r2.acts[i])
                maxes1 = maxindex(r1copy, 5)
                maxes2 = maxindex(r2copy, 5)

                mask1 = np.ones(len(r1copy), np.bool)
                mask1[ints(maxes1)] = 0

                # mask2 = np.ones(len(r2copy), np.bool)
                # mask2[ints(maxes2)] = 0

                # mask = np.bitwise_and(mask1, mask2)

                # r1copy[mask1] = np.nan
                # r2copy[mask1] = np.nan

                # r1copy[mask1] = None
                # r2copy[mask1] = None

                r1copy = r1copy[ints(maxes1)]
                r2copy = r2copy[ints(maxes1)]

                # zs = np.bitwise_or(r1copy == 0, r2copy == 0)
                if all(r1copy == 0) or all(r2copy == 0):
                    r += np.nan
                else:
                    r += scipy.stats.spearmanr(r1copy, r2copy, nan_policy='omit')[0]
            return r
        return self.confuse_analysis(
            data,
            spearman_r,
            identiy=False
        )

    @shadow(ftype=ShadowFigType.RAW)
    @cell(inputs=spearman_r_cmat)
    def spearman_r_texttable(self, cmat):
        def int_if_not_none_or_str(e):
            if isnan(e) or isstr(e):
                return e
            else:
                return int(e)
        return self.cmat_html_table(
            cmat,
            vectorfun=np.vectorize(int_if_not_none_or_str),
            int_thresh=90
        )

    @cell(inputs=calc_accs)
    def save(self, data):
        File(data['dest']).save(data)




    def get_report_figdata(self, exp_name, resources_root, database):
        data = resources_root['sanity.pickle'].load()
        return self.acc_table(data).children
