from complexity_considerations.model_size import get_keras_model_size
from complexity_considerations.convert_to_1bit import convert_to_1bit_conv, set_to_1bit
import config
from models import construct_model
from load_data import load_h5s


def get_model_size(model_path):
    get_keras_model_size(model_path)


if __name__ == '__main__':
    x, y, val_x, val_y = load_h5s(config.home_path, config.data_path, config.validation_file, config.training_file)

    model = construct_model(x, y)

    folder2store = '/home/javi/repos/DCASE2021-Task1/outputs/2020-12-27-10:48/'
    model_name = 'best.h5'

    model.load_weights(folder2store + model_name)

    print(model.summary())
    get_model_size(model)
    convert_to_1bit_conv(model, folder2store)
    converted_model = set_to_1bit(model, folder2store)
    #TODO: model size do not change
    get_model_size(converted_model)
