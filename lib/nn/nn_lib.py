import copy

from lib.figs.TableData import RSAMatrix
from lib.defaults import *
from lib.misc.mutil import arr, itr, zeros
from lib.nn.nn_plotting import sort_human
from lib.data_saving import savePlotAndTableData

def calc_steps(N_IMAGES, TRAIN_TEST_SPLIT, BATCH_SIZE):
    N_TRAIN_IMAGES = int(N_IMAGES * TRAIN_TEST_SPLIT)
    N_TEST_IMAGES = int(N_IMAGES * (1 - TRAIN_TEST_SPLIT))
    TRAIN_STEPS = int(N_TRAIN_IMAGES / BATCH_SIZE)
    TEST_STEPS = int(N_TEST_IMAGES / BATCH_SIZE)
    return TRAIN_STEPS, TEST_STEPS, N_TRAIN_IMAGES, N_TEST_IMAGES

def symm(im, arg):
    ODD = mod(im.shape[1], 2) > 0
    # err('please use images with an even number of columns (will fix later)')
    n_sym = 0
    sym_i = []
    n_pix = numel(im)
    if ODD:
        left_pix = (n_pix - im.shape[0]) / 2
    else:
        left_pix = n_pix / 2
    left_pix = int(left_pix)
    i = 0
    for x in range(im.shape[1]):
        if x < im.shape[1] / 2:
            x2 = im.shape[1] - (x + 1)
            for y in range(im.shape[0]):
                i = i + 1
                this_sym = (im[y, x] == im[y, x2])
                n_sym = n_sym + this_sym
                if this_sym:
                    sym_i.append(i)

    if arg is None:
        pass
        # what was that for? sym_im was unused
        # sym_im = n_sym / left_pix
    else:
        p_sym = arg
        n_sym_target = round(left_pix * p_sym)
        sym_i_target = copy.deepcopy(sym_i)

        #     get all sym_i_target
        more = n_sym_target - len(sym_i)
        more_get_i = more
        #     while more_get_i > 0
        new_order = randperm(left_pix)
    i = 0
    for new in new_order:
        i = i + 1
        if new not in sym_i:  # and not ismember(new, sym_i_target):
            # log('here4.2')
            sym_i_target.append(new)
            more_get_i = more_get_i - 1
            if more_get_i == 0:
                # log('here4.3')
                break

    sym_im = im
    i = 0
    for x in range(left_pix):
        if x < im.shape[1] / 2:
            x2 = im.shape[1] - (x + 1)
            for y in range(im.shape[0]):
                i = i + 1
                if not ismember(i, sym_i) and ismember(i, sym_i_target):
                    sym_im[y, x2] = im[y, x]

    return sym_im

# layers = freezeWeights(layers) sets the learning rates of all the
# parameters of the layers in the layer array |layers| to zero.
# def freezeWeights(layers):
#     for ii in layers.shape[0]:
#         err('needs work')
#         props = properties(layers(ii))
#         for p in range(numel(props)):
#             propName = props[p]
#             if not isempty(regexp(propName, 'LearnRateFactor$', 'once')):
#                 # layers[ii].__setattr__()
#                 setattr(layers[ii], propName, 0)
#                 # layers[ii].propName = 0
#
#     return layers

# findLayersToReplace(lgraph) finds the single classification layer and the
# preceding learnable (fully connected or convolutional) layer of the layer
# graph lgraph.
# def findLayersToReplace(lgraph):
#     # if not isa(lgraph,'nnet.cnn.LayerGraph'):
#     #     err('Argument must be a LayerGraph object.')
#     # Get source, destination, and layer names.
#     src = str(lgraph.Connections.Source)
#     dst = str(lgraph.Connections.Destination)
#
#     # this line of code may not have translated well from matlab to python
#     layerNames = str(lgraph.Layers.arrayof().Name)
#
#     # Find the classification layer. The layer graph must have a single
#     # classification layer.
#     isClassificationLayer = arrayfun(lambda l: \
#                                          (isa(l, 'nnet.cnn.layer.ClassificationOutputLayer') | isa(l,
#                                                                                                    'nnet.layer.ClassificationLayer')), \
#                                      lgraph.Layers)
#
#     if sum(isClassificationLayer) != 1:
#         err('Layer graph must have a single classification layer.')
#     classLayer = lgraph.Layers(isClassificationLayer)
#
#     # Traverse the layer graph in reverse starting from the classification
#     # layer. If the network branches, throw an error.
#     currentLayerIdx = find(isClassificationLayer)
#     while True:
#         if numel(currentLayerIdx) != 1:
#             err('Layer graph must have a single learnable layer preceding the classification layer.')
#
#         currentLayerType = classname(lgraph.Layers(currentLayerIdx))
#         isLearnableLayer = ismember(currentLayerType, \
#                                     ['nnet.cnn.layer.FullyConnectedLayer', 'nnet.cnn.layer.Convolution2DLayer'])
#
#         if isLearnableLayer:
#             learnableLayer = lgraph.Layers(currentLayerIdx)
#             return learnableLayer, classLayer
#
#         currentDstIdx = find(layerNames(currentLayerIdx) == dst)
#         currentLayerIdx = find(src(currentDstIdx) == layerNames)

# lgraph = createLgraphUsingConnections(layers,connections) creates a layer
# graph with the layers in the layer array |layers| connected by the
# connections in |connections|.

# def createLgraphUsingConnections(layers, connections):
#     lgraph = layerGraph()
#     for i in range(layers.size):
#         lgraph = addLayers(lgraph, layers(i))
#
#     for c in range(connections.shape[0]):
#         lgraph = connectLayers(lgraph, connections.Source[c], connections.Destination[c])
#
#     return lgraph

def RSA(nam, rep, y_true, ei, layer_name=None, layer_i=None):
    log('running RSA...')
    special_confuse_mat = zeros(len(rep), len(rep))
    classes = []

    inter_activations = rep.tolist()
    from lib.nn import nnstate
    TEST_CLASS_MAP = nnstate.CURRENT_TRUE_MAP

    alpha_keys = sort_human(list(TEST_CLASS_MAP.keys()))

    acts_trues = [(inter_activations[i], y_true[i]) for i in itr(y_true)]

    # class_names =list(test_classes.keys())
    # sorted_acts = sort(acts_trues,key=lambda t: t[1])
    # matching darius' alphabatized order
    class_names = alpha_keys
    sorted_acts = sort_human(acts_trues, keyparam=lambda t: inv_map(TEST_CLASS_MAP)[t[1]])

    sorted_acts = arr([a[0] for a in sorted_acts])

    log('getting norms...')
    with Progress(len(sorted_acts)) as prog:
        for i in itr(sorted_acts):
            classes.append(list(TEST_CLASS_MAP.keys())[y_true[i]])
            for j in itr(sorted_acts):
                norm = np.linalg.norm(sorted_acts[i, :] - sorted_acts[j, :])
                special_confuse_mat[i, j] = norm
            prog.tick()
    log('finished getting norms!')

    mx = np.max(special_confuse_mat)

    def fix(thing):
        return abs(mx - thing)

    special_confuse_mat = np.vectorize(fix)(special_confuse_mat)

    tit = f'L2-{nam}'
    title = f'{tit} ({nnstate.FLAGS.arch}{nnstate.FLAGS.ntrain}E{ei + 1})'
    if nam == 'Inter':
        title = f'{title}(Layer{layer_i}:{layer_name})'

    breakpoint()
    savePlotAndTableData(RSAMatrix(
        data=special_confuse_mat,
        title=title,
        confuse_max=1,
        confuse_target=mx,
        block_len=10,
        row_headers=class_names,
        col_headers=class_names,
    ), tit, f'CM{ei + 1}')
