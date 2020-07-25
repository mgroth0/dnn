from lib.dnn_analyses import PerEpochAnalysis
from lib.dnn_data_saving import save_dnn_data
from lib.dnn_proj_struct import experiments_from_folder
from mlib.boot import log
from mlib.boot.lang import enum
class ExampleInput(PerEpochAnalysis):

    @staticmethod
    def _after_thing(datagen, nam):
        log('saving examples')
        exs = datagen.examples()
        for idx, ex in enum(exs):
            save_dnn_data(ex[1], nam, ex[0], 'png')
        log('finished saving examples')


    def after_val(self, i, net, nam):
        if i == 0: self._after_thing(net.val_data, nam)
    def after_fit(self, i, net, nam):
        if i == 0: self._after_thing(net.train_data, nam)
    def during_compile(self, eg):
        breakpoint()
        experiments = experiments_from_folder(eg.folder)
        if len(experiments) > 0: # might be zero if only did sanity test
            random_exp = experiments[0]
            for phase in ['train', 'val']:
                for ex in random_exp.folder[f'{phase}'].safeglob('*.png'):
                    ex.copy_to(
                        eg.compile_exp_res_folder[f'examples_{phase}/{ex.name_pre_ext}.png']
                    )
    def get_report_figdata(self, exp_name, resources_root, database):
        all_examples = []
        for example_im in resources_root.glob('example*/*.png'):
            single_file = example_im.copy_to(
                resources_root[f'_examples/{example_im.parent.name.replace("examples_", "")}-{example_im.name}']
            )
            all_examples += [(
                single_file,
                database.get_or_set_default(
                    '',
                    exp_name,
                    single_file.name_pre_ext
                )
            )]
        return all_examples
