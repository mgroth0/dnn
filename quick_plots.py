import matplotlib.pyplot as plt

from files import SALIENCE_RESULT_FOLDER
from mlib.boot.lang import listkeys
from mlib.str import shorten_str

DEBUG_REMAKE = False

def main():
    print('hello world')

    # models = {
    #     'keras_no_bn'  : 'keras_1612270680',
    #     'pytorch_no_bn': 'pytorch_1612268175',
    #     'pytorch_bn'   : 'pytorch_1612275289'
    # }
    models = {
        # 'keras_no_bn_more_ims': 'keras_1612790199',
        # 'keras_no_bn_more_ims2': 'keras_1612887039',

        # 'pytorch_bn2'   : 'pytorch_bn_1612415904',
        # 'pytorch_zoo_1'   : 'pytorch_zoo_1612889472'
        # 'pytorch_zoo_50'   : 'pytorch_zoo_1612891265'
        # 'keras_no_bn_more_ims_combined': 'keras_combined',


        # 'keras_no_bn_more_ims_combined': 'keras_combined',

        'keras_new': 'keras_1614492790',
    }

    zoos = ['pytorch_zoo_1', 'pytorch_zoo_50']

    # fig_root = '_figs/salience'
    # fig_root = '_data/result/salience_figs'

    models = {}
    i = 0

    for folder in SALIENCE_RESULT_FOLDER.files:
        if folder.name == 'old': continue
        # if folder.name == 'salience_figs': continue
        models[f'keras{i}'] = folder.name
        i = i + 1

    for model in listkeys(models):
        # with open(f'tf_bug1/data_tfbug/data_result/{models[model]}/data_result.json', 'r') as f:
        my_result_folder = SALIENCE_RESULT_FOLDER[models[model]]
        my_fig_root = my_result_folder['figs']
        if my_fig_root.exists and not (DEBUG_REMAKE):
            print(f'not making plot for {models[model]}, already exists')
        else:
            print(f'making plot for {models[model]}, does not yet exist')
            my_fig_root.mkdirs()
            data = my_result_folder['data_result.json'].load()
            log_data = my_result_folder['log.pkl'].load()
            _main_plot(data, model, zoos, my_fig_root)
            _log_plot(log_data, my_fig_root, model)
    print('done')
def _main_plot(data, model, zoos, fig_root):
    factor = 1 if 'pytorch' in model else 100

    num_imss = []
    total_averages = []
    last_10_averages = []
    lasts = []

    for i in range(len(data)):  # for num_ims/model

        num_imss.append(data[i]["num_images"])
        total_averages.append(sum(data[i]['history']['accuracy']) / len(data[i]['history']['accuracy']))
        last_10_averages.append(sum(data[i]['history']['accuracy'][-10:]) / 10)
        lasts.append([-1])

        plt.plot(
            [x * factor for x in data[i]['history']['accuracy']], 'b', label='Training Accuracy'
        )
        plt.plot(
            [x * factor for x in data[i]['history']['val_accuracy']], 'g', label='Evaluation Accuracy'
        )
        plt.legend()
        plt.ylim(0, 100)
        num_images = data[i]["num_images"]
        if model not in zoos:
            plt.title(f'{model},num_images={num_images}')
        else:
            plt.title(f'{data[i]["model_name"]},num_images={num_images}')
        print('saving figure')

        if model not in zoos:
            plt.savefig(fig_root[f'{data[i]["num_images"]}.png'].abspath)
        else:
            plt.savefig(fig_root[f'{data[i]["model_name"]}.png'].abspath)
        plt.clf()

    if model not in zoos:
        plt.plot(num_imss, total_averages, 'b', label='total average accuracy')
        plt.plot(num_imss, last_10_averages, 'g', label='average accuracy last 10 epochs')
        plt.plot(num_imss, lasts, 'r', label='average accuracy of final epoch')
        plt.title("how results vary as we modify # training images")
        plt.xlabel("# training and testing images per epoch")
        plt.legend()
        plt.ylim(0, 1)

        plt.savefig(fig_root['summary.png'].abspath)
        plt.clf()




if __name__ == '__main__':
    main()
