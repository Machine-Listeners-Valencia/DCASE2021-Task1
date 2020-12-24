from complexity_considerations.model_size import get_keras_model_size
from complexity_considerations.convert_to_1bit import convert_to_1bit_conv
from keras.models import load_model
from focal_loss import categorical_focal_loss
import dill
import config


def get_model_size(model_path):
    get_keras_model_size(model_path)


if __name__ == '__main__':
    model_path = '/home/javi/repos/DCASE2021-Task1/outputs/2020-12-23-16:09/best.h5'

    n_classes = 10

    if config.loss_type == 'focal_loss':
        if type(config.fl_alpha) is not list:
            alpha_list = [[config.fl_alpha] * n_classes]
        else:
            alpha_list = config.fl_alpha
            # check_alpha_list(alpha_list, y.shape[1])

    custom_object = {'categorical_focal_loss_fixed': dill.loads(
        dill.dumps(categorical_focal_loss(gamma=config.fl_gamma, alpha=alpha_list)))}

    model = load_model(model_path, custom_objects=custom_object)
    print(model.summary())
    get_model_size(model)
    convert_to_1bit_conv(model)
