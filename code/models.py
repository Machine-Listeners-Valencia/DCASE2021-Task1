import keras.layers
from modules import network_module

from keras.models import Model


def res_conv_standard_post_csse(h, w, n_channels, n_classes,
                                nfilters, pools_size, dropouts_rate, ratio, reshape_type, dense_layer,
                                verbose=False):
    """
    Model
    """

    ip = keras.layers.Input(shape=(h, w, n_channels))

    for i in range(0, len(nfilters)):

        if i == 0:
            x = network_module(ip, nfilters[i], ratio, pools_size[i], dropouts_rate[i])

        else:
            x = network_module(x, nfilters[i], ratio, pools_size[i], dropouts_rate[i])

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

#TODO trident model