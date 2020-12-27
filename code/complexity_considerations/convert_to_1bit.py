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
