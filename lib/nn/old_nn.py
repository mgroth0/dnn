from loggy import log
from data_saving import savePlotAndTableData

def old(net):
    fs = FigSet()
    filetypes = [
        'jpeg',
        'jpeg',
        'jpeg',
        'jpeg',
        'jpeg',
        'jpeg',
        'png',
        'jpeg',
        'jpeg',
        'jpeg'
    ]

    for i in range(10):
        fs.viss.clear()
        fs.viss.append(FigData())
        fs.viss[0].item_type = 'image'

        # Read the image to classify
        file = 'images/' + str(i + 1) + '.' + filetypes[i]
        log('working with file:$', file)
        if filetypes[i] == 'png':
            I = imread(file)
            I = make255(I)
        else:
            I = imread(file).astype(np.uint8)
        # Adjust size of the image
        if not JUST_TEST_SMALL_IMAGES:
            sz = net.layers[0]._batch_input_shape  # [1:3]
        else:
            sz = (3, 3, 3)

        if RESIZE:
            I = arr(Image.fromarray(I).resize((sz[1], sz[2])).getdata())

            if JUST_TEST_SMALL_IMAGES:
                I = np.resize(I, (3, 3, 3))
            else:
                I = np.resize(I, (299, 299, 3))


        else:
            I = I[0:sz[1], 0:sz[2], 0:sz[3]]

        # for ii in range(I.shape[1]):
        #     I[0,ii] = I[0,ii].insert(0)
        #     # Classify the image using GoogLeNet
        # for
        # make valus from 0 to 1
        I = make1(I)

        if JUST_TEST_SMALL_IMAGES:
            label = 'test'
        else:
            label = net.predict(arr([I]))
            label = maxindex(label)
            label = irn2.ImageNetClasses[label]
        log('label:$', label)
        #     # Show the image and the classification results
        fs.viss[0].x = I.tolist()

        title = str(i + 1) + ':' + label
        fs.viss[0].title = title

        savePlotAndTableData(fs, 'dnn-old', title.replace(':', '-'))

if __name__ == '__main__':
    net = SymmetryNet()
    # net.net.summary() # See details of the architecture
    old(net.net)