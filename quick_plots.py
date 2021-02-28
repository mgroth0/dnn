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


def _log_plot(log_data, fig_root, model):
    important_text = []
    important_time = []
    for lin, file_line, t in log_data:
        if 'bash dnn.simgw' in file_line:
            important_text.append('bash dnn.simgw')
            important_time.append(t)
        if '__DNN_IS_FINISHED__' in file_line and 'got __DNN_IS_FINISHED__' not in file_line:
            important_text.append('__DNN_IS_FINISHED__')
            important_time.append(t)

    important_text = [shorten_str(s, 20) for s in important_text]

    fig, axs = plt.subplots(nrows=2)
    table_ax = axs[0]
    pie_ax = axs[1]
    table_ax.set_axis_off()

    important_time = [round(t, 2) for t in important_time]
    table = table_ax.table(
        cellText=[[str(t)] for t in important_time],
        rowLabels=important_text,
        colLabels=['time'],
        rowColours=["palegreen"] * (len(important_text) + 1),
        colColours=["palegreen"] * 2,
        colWidths=[0.5, 0.5],
        cellLoc='center',
        loc='center'
    )

    table_ax.set_title('Important Logs', fontweight="bold")

    time_amounts = []
    last = 0
    for t in important_time:
        time_amounts.append(t - last)
        last = t
    sizes = important_time
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

    while len(colors) < len(sizes):
        colors = colors + colors
    colors = colors[:len(sizes)]

    pie = pie_ax.pie(
        time_amounts,
        labels=important_text,
        colors=colors
    )

    pie_ax.text(
        0.5,
        -0.25,
        '* +2 seconds for figure generation',
        horizontalalignment='center',
        verticalalignment='center',
        transform=pie_ax.transAxes
    )

    plt.savefig(fig_root['logs.png'].abspath)

    plt.clf()

if __name__ == '__main__':
    main()
