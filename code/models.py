import keras.layers
from modules import network_module, freq_split

from keras.models import Model


__authors__ = "Javier Naranjo, Sergi Perez and Irene Martín"
__copyright__ = "Machine Listeners Valencia"
__credits__ = ["Machine Listeners Valencia"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Javier Naranjo"
__email__ = "janal2@alumni.uv.es"
__status__ = "Dev"
__date__ = "2020"


def res_conv_standard_post_csse(h, w, n_channels, n_classes,
                                nfilters, pools_size, dropouts_rate, ratio, reshape_type, dense_layer,
                                pre_act=False, shortcut='conv', verbose=False):
    """
    Model
    """

    ip = keras.layers.Input(shape=(h, w, n_channels))

    for i in range(0, len(nfilters)):

        if i == 0:
            x = network_module(ip, nfilters[i], ratio, pools_size[i], dropouts_rate[i], i,
                               pre_act=pre_act, shortcut=shortcut)

        else:
            x = network_module(x, nfilters[i], ratio, pools_size[i], dropouts_rate[i], i,
                               pre_act=pre_act, shortcut=shortcut)

    # Reshape
    if reshape_type == 'global_avg':
        x = keras.layers.GlobalAveragePooling2D()(x)

    elif reshape_type == 'flatten':
        x = keras.layers.Flatten()(x)

    elif reshape_type == 'global_max':
        x = keras.layers.GlobalMaxPooling2D()(x)

    if dense_layer is None:
        x = keras.layers.Dense(n_classes)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('softmax')(x)

    model = Model(ip, x)

    if verbose:
        print(model.summary())

    return model


def res_conv_standard_post_csse_split_freqs(h, w, n_channels, n_classes,
                                            nfilters, pools_size, dropouts_rate, ratio, reshape_type, dense_layer,
                                            n_split_freqs, f_split_freqs, pre_act=False, shortcut='conv',
                                            verbose=False):
    ip = keras.layers.Input(shape=(h, w, n_channels))

    if n_split_freqs == 2:

        splits = keras.layers.Lambda(freq_split, arguments={'n_split_freqs': n_split_freqs,
                                                            'f_split_freqs': f_split_freqs})(ip)

        x1 = splits[0]
        x2 = splits[1]

        for i in range(0, len(nfilters)):
            x1 = network_module(x1, nfilters[i], ratio, pools_size[i], i, dropouts_rate[i],
                                pre_act=pre_act, shortcut=shortcut)
            x2 = network_module(x2, nfilters[i], ratio, pools_size[i], i, dropouts_rate[i],
                                pre_act=pre_act, shortcut=shortcut)

        x = keras.layers.concatenate([x1, x2], axis=1)

        # Reshape
        if reshape_type == 'global_avg':
            x = keras.layers.GlobalAveragePooling2D()(x)

        elif reshape_type == 'flatten':
            x = keras.layers.Flatten()(x)

        elif reshape_type == 'global_max':
            x = keras.layers.GlobalMaxPooling2D()(x)

        if dense_layer is None:
            x = keras.layers.Dense(n_classes)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('softmax')(x)

        model = Model(ip, x)

        if verbose:
            print(model.summary())

        return model

    elif n_split_freqs == 3:

        splits = keras.layers.Lambda(freq_split, arguments={'n_split_freqs': n_split_freqs,
                                                            'f_split_freqs': f_split_freqs})(ip)

        x1 = splits[0]
        x2 = splits[1]
        x3 = splits[2]

        for i in range(0, len(nfilters)):
            x1 = network_module(x1, nfilters[i], ratio, pools_size[i], i, dropouts_rate[i],
                                pre_act=pre_act, shortcut=shortcut)
            x2 = network_module(x2, nfilters[i], ratio, pools_size[i], i, dropouts_rate[i],
                                pre_act=pre_act, shortcut=shortcut)
            x3 = network_module(x3, nfilters[i], ratio, pools_size[i], i, dropouts_rate[i],
                                pre_act=pre_act, shortcut=shortcut)

        x = keras.layers.concatenate([x1, x2, x3], axis=1)

        # Reshape
        if reshape_type == 'global_avg':
            x = keras.layers.GlobalAveragePooling2D()(x)
        elif reshape_type == 'flatten':
            x = keras.layers.Flatten()(x)
        elif reshape_type == 'global_max':
            x = keras.layers.GlobalMaxPooling2D()(x)

        if dense_layer is None:
            x = keras.layers.Dense(n_classes)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('softmax')(x)

        model = Model(ip, x)

        if verbose:
            print(model.summary())

        return model
