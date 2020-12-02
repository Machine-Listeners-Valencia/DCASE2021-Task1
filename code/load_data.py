import h5py
import numpy as np


def load_h5s():
    val_hdf5_path = '' # TODO: adding files

    hf = h5py.File(val_hdf5_path, 'r')

    val_x = hf['features'][:]
    val_x = np.expand_dims(val_x, axis=-1)

    val_gamma_y = hf['labels'][:]
    hf.close()

    train_hdf5_path = '' # TODO: addig files

    hft = h5py.File(train_hdf5_path, 'r')
    train_x = hft['features'][:]
    train_x = np.expand_dims(train_x, axis=-1)

    train_gamma_y = hft['labels'][:]

    hft.close()

    return train_x, val_x
