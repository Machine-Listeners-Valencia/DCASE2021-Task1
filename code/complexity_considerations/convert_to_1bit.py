import numpy as np
import code.config
import code.focal_loss
import dill
from keras.models import load_model


def convert_to_1bit_conv(model):
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
            ZeroOneWeightsDict[layer.name] = binary_weights
            AllParamsDict[layer.name] = binary_weights
            NumBinaryWeights += np.prod(ww[0].shape)

        elif 'bn' in layer.name:
            # the saved model also nees floating point batch norm params
            ww = layer.get_weights()
            AllParamsDict[layer.name] = ww
            cc = 0
            for kk in ww:
                # print(cc,layer.name,np.prod(kk.shape))
                Num32bitWeights += np.prod(kk.shape)
                cc = cc + 1

    # savemat('FinalModel_01weights.mat' ,ZeroOneWeightsDict ,do_compression=True ,long_field_names=True)
    # savemat('FinalModel_allparams.mat' ,AllParamsDict ,do_compression=True ,long_field_names=True)

    WeightsMemory = NumBinaryWeights / 8 / 1024
    BNMemory = 32.0 * Num32bitWeights / 8 / 1024
    print('Num binary weights is less than 500kb: ', int(NumBinaryWeights), 'conv weights = conv weights memory of '
          , WeightsMemory, '  kB')
    print('Num 32-bit weights (all batch norm parameters) = ', int(Num32bitWeights), '; weights memory = ', BNMemory
          , '  kB')
    print('Total memory = ', WeightsMemory + BNMemory, '  MB')


if __name__ == '__main__':
    model_path = '/home/javi/repos/DCASE2021-Task1/outputs/2020-12-23-16:09/best.h5'

    n_classes = 10

    if code.config.loss_type == 'focal_loss':
        if type(code.config.fl_alpha) is not list:
            alpha_list = [[code.config.fl_alpha] * n_classes]
        else:
            alpha_list = code.config.fl_alpha
            # check_alpha_list(alpha_list, y.shape[1])

    custom_object = {'categorical_focal_loss_fixed': dill.loads(
        dill.dumps(code.focal_loss.categorical_focal_loss(gamma=code.config.fl_gamma, alpha=alpha_list)))}

    model = load_model(model_path, custom_objects=custom_object)
    convert_to_1bit_conv(model)
