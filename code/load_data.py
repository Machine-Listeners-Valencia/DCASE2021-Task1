import h5py
import numpy as np
import os

__authors__ = "Javier Naranjo, Sergi Perez and Irene Martín"
__copyright__ = "Machine Listeners Valencia"
__credits__ = ["Machine Listeners Valencia"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Javier Naranjo"
__email__ = "janal2@alumni.uv.es"
__status__ = "Dev"
__date__ = "2020"


def load_h5s(path_to_data):
    home = os.getenv('HOME')

    val_hdf5_path = home + path_to_data + 'train_val_gammatone_mono_f2.h5'  # TODO: use pathlib

    hf = h5py.File(val_hdf5_path, 'r')

    val_x = hf['features'][:]
    val_x = np.expand_dims(val_x, axis=-1)

    val_y = hf['labels'][:]
    hf.close()

    train_hdf5_path = home + path_to_data + 'train_val_gammatone_mono_f1.h5'  # TODO: use pathlib

    hft = h5py.File(train_hdf5_path, 'r')
    train_x = hft['features'][:]
    train_x = np.expand_dims(train_x, axis=-1)

    train_y = hft['labels'][:]

    hft.close()

    return train_x, train_y, val_x, val_y
