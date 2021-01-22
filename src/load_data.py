import h5py
import numpy as np
import os

__authors__ = "Javier Naranjo, Sergi Perez and Irene Martín"
__copyright__ = "Machine Listeners Valencia"
__credits__ = ["Machine Listeners Valencia"]
__license__ = "MIT License"
__version__ = "0.2.0"
__maintainer__ = "Javier Naranjo"
__email__ = "janal2@alumni.uv.es"
__status__ = "Dev"
__date__ = "2020"


def load_h5s(home_path, path_to_data, validation_file, training_file):
    """
    Function that reads mono h5 files and return training and validation features
    :param home_path: home path where the project is stored
    :param path_to_data: path to folder where h5 are stored
    :param validation_file: name of the validation h5 file
    :param training_file: name of the training h5 file
    :return: features and one-hot encodings for training and validation
    """
    if home_path is None:
        home = os.getenv('HOME')

    else:
        home = home_path

    val_hdf5_path = home + path_to_data + validation_file  # TODO: use pathlib

    hf = h5py.File(val_hdf5_path, 'r')

    val_x = hf['features'][:]
    val_x = np.expand_dims(val_x, axis=-1)

    val_y = hf['labels'][:]
    hf.close()

    train_hdf5_path = home + path_to_data + training_file  # TODO: use pathlib

    hft = h5py.File(train_hdf5_path, 'r')
    train_x = hft['features'][:]
    train_x = np.expand_dims(train_x, axis=-1)

    train_y = hft['labels'][:]

    hft.close()

    return train_x, train_y, val_x, val_y
