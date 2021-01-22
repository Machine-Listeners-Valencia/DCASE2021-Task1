import numpy as np
from scipy.io import savemat, loadmat


# TODO: it does not work with regular conv2d layer
def convert_to_1bit_conv(model, folder2store):
    ZeroOneWeightsDict = {}
    AllParamsDict = {}
    NumBinaryWeights = 0.0
    Num32bitWeights = 0.0
    for layer in model.layers:
        # print(layer.name)

        if 'conv' in layer.name:
            ww = layer.get_weights()

            # storage using 1 bit booleans
            binary_weights = (0.5 * (np.sign(ww) + 1.0)).astype('bool')  # save weights as 0 or 1

            # The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
            # binary_weights = (0.5 * (np.sign(ww.all()) + 1.0)).astype('bool')  # save weights as 0 or 1
            ZeroOneWeightsDict[layer.name] = binary_weights
            AllParamsDict[layer.name] = binary_weights
            NumBinaryWeights += np.prod(ww[0].shape)

        elif 'bn' in layer.name:
            # the saved model also needs floating point batch norm params
            ww = layer.get_weights()
            AllParamsDict[layer.name] = ww
            cc = 0
            for kk in ww:
                # print(cc,layer.name,np.prod(kk.shape))
                Num32bitWeights += np.prod(kk.shape)
                cc = cc + 1

    savemat(folder2store + 'FinalModel_01weights.mat', ZeroOneWeightsDict, do_compression=True, long_field_names=True)
    savemat(folder2store + 'FinalModel_allparams.mat', AllParamsDict, do_compression=True, long_field_names=True)

    WeightsMemory = NumBinaryWeights / 8 / 1024
    BNMemory = 32.0 * Num32bitWeights / 8 / 1024
    print('Num binary weights is less than 500kb: ', int(NumBinaryWeights), 'conv weights = conv weights memory of '
          , WeightsMemory, '  kB')
    print('Num 32-bit weights (all batch norm parameters) = ', int(Num32bitWeights), '; weights memory = ', BNMemory
          , '  kB')
    print('Total memory = ', WeightsMemory + BNMemory, '  MB')


def set_to_1bit(model, folder2store):
    AllParamsDict_loaded = loadmat(folder2store + 'FinalModel_allparams.mat')

    conv_names = [m for m in list(AllParamsDict_loaded.keys()) if any(s in m for s in ['conv'])]
    bn_names = [m for m in list(AllParamsDict_loaded.keys()) if any(s in m for s in ['bn'])]

    c1 = 0
    c2 = 0
    for layer in model.layers:
        if 'conv' in layer.name:
            ww = AllParamsDict_loaded[conv_names[c1]].astype('float32') * 2.0 - 1.0
            ww = ww * np.sqrt(2.0 / np.prod(ww[0].shape[0:3]))
            layer.set_weights([ww[0]])
            print('conv layer ', c1, ' has ', len(np.unique(ww)), ' unique weight values')
            c1 = c1 + 1
        elif 'bn' in layer.name:
            ww = AllParamsDict_loaded[bn_names[c2]]
            layer.set_weights(ww)
            c2 = c2 + 1

    return model
