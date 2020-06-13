from lib.boot.loggy import log
from lib.misc.mutil import File, Temp
from lib.nn.gen_preproc_ims import SymClassGroup, gen_images, get_class_dict

def nn_gen(FLAGS, folder,
           num_gpus=4,
           TRAINING_SET_SIZES=(25, 50, 100, 150, 200, 1000),
           EVAL_SIZE=500,
           RSA_SIZE_PER_CLASS=10
           ):
    import lib.nn.nnstate as nnstate
    nnstate.FLAGS = FLAGS

    # TEST_BANDS = [0,2,4,6,-4,]

    TEST_BANDS = [
        SymClassGroup(0, False),
        SymClassGroup(2, False),
        SymClassGroup(4, False),
        SymClassGroup(6, False),
        SymClassGroup(8, False),
        SymClassGroup(10, False),
        SymClassGroup(0, True),
        SymClassGroup(2, True),
        SymClassGroup(4, True),
        SymClassGroup(6, True),
        SymClassGroup(8, True),
        SymClassGroup(10, True)
    ]

    # band_groups = [[0], [4], [0, 4],[0,2,4,6],[-4]]
    bands = [
        SymClassGroup(0, False),
        SymClassGroup(4, False)
    ]

    # band_groups = [[0, 4]]
    # cols = [[0, 0, 1], [1, 0, 0], [1, 0, 1],[0,1,0],[0.5,0.5,0.5]]
    cols = [[1, 0, 1]]
    # nams = ['nb', 'b4', 'nb_b4','0246','d4']
    nams = ['nb_b4']

    # if max(map(len, band_groups)) > 1: MULTI = True
    NUM_PAWAN_TEST_IMAGES = RSA_SIZE_PER_CLASS * len(TEST_BANDS) * 2

    # TRAIN_STEPS, TEST_STEPS, N_TRAIN_IMAGES, N_TEST_IMAGES = calc_steps(N_IMAGES, TRAIN_TEST_SPLIT, BATCH_SIZE)
    # PAWAN_TRAIN_STEPS, PAWAN_TEST_STEPS, PAWAN_N_TRAIN_IMAGES, PAWAN_N_TEST_IMAGES = calc_steps(
    #     NUM_PAWAN_TEST_IMAGES, 0, nn_main.BATCH_SIZE)
    test_classes = get_class_dict(TEST_BANDS)

    File(folder).deleteIfExists()
    folder.mkdirs()
    folder = File(folder)

    rsaFold = folder.resolve('ForMatt')
    testFold = folder.resolve('Testing')
    trainFold = folder.resolve('Training')

    gen_images(NUM_PAWAN_TEST_IMAGES, test_classes, bands=TEST_BANDS, folder=rsaFold)
    gen_images(EVAL_SIZE * len(TEST_BANDS) * 2 ,             test_classes, bands=TEST_BANDS, folder=testFold)

    classes = get_class_dict(bands=bands)
    for n in TRAINING_SET_SIZES:
        size = n * 2 * len(bands)
        gen_images(size, classes, bands=bands, folder=trainFold.resolve(str(n)))

    import shutil
    with Temp('_temp_ims') as f:
        for i in range(num_gpus):
            log('copytree...')
            shutil.copytree(folder, f.abspath + '/gpu' + str(i + 1))
        for i in range(num_gpus):
            log('moving...')
            shutil.move(f.abspath + '/gpu' + str(i + 1), folder.abspath)
    rsaFold.delete()
    testFold.delete()
    trainFold.delete()
