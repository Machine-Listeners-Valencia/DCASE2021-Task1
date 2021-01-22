from modules import network_module, freq_split
import config

if config.tf:
    from tensorflow.keras.layers import (Dense, BatchNormalization, Activation, GlobalAveragePooling2D,
                                         Flatten, GlobalMaxPooling2D, Input, Lambda, concatenate)
    from tensorflow.keras.models import Model
else:
    from keras.layers import (Dense, BatchNormalization, Activation, GlobalAveragePooling2D,
                              Flatten, GlobalMaxPooling2D, Input, Lambda, concatenate)
    from keras.models import Model

__authors__ = "Javier Naranjo, Sergi Perez and Irene Martín"
__copyright__ = "Machine Listeners Valencia"
__credits__ = ["Machine Listeners Valencia"]
__license__ = "MIT License"
__version__ = "0.4.0"
__maintainer__ = "Javier Naranjo"
__email__ = "janal2@alumni.uv.es"
__status__ = "Dev"
__date__ = "2020"


def construct_model(x, y):
    if config.split_freqs is not True:
        model = res_conv_standard_post_csse(x.shape[1], x.shape[2], x.shape[3], y.shape[1],
                                            config.n_filters, config.pools_size, config.dropouts_rate, config.ratio,
                                            config.reshape_method, config.dense_layer,
                                            pre_act=config.pre_act, shortcut=config.shortcut, verbose=config.verbose,
                                            binary_layer=config.binary_layer)

    else:
        model = res_conv_standard_post_csse_split_freqs(x.shape[1], x.shape[2], x.shape[3], y.shape[1],
                                                        config.n_filters, config.pools_size, config.dropouts_rate,
                                                        config.ratio,
                                                        config.reshape_method, config.dense_layer,
                                                        config.n_split_freqs, config.f_split_freqs,
                                                        pre_act=config.pre_act, shortcut=config.shortcut,
                                                        verbose=config.verbose, binary_layer=config.binary_layer)

    return model


def res_conv_standard_post_csse(h, w, n_channels, n_classes,
                                nfilters, pools_size, dropouts_rate, ratio, reshape_type, dense_layer,
                                pre_act=False, shortcut='conv', verbose=False, binary_layer=False):
    """
    Model
    """

    ip = Input(shape=(h, w, n_channels))

    for i in range(0, len(nfilters)):

        if i == 0:
            x = network_module(ip, nfilters[i], ratio, pools_size[i], dropouts_rate[i], i,
                               pre_act=pre_act, shortcut=shortcut, binary_layer=binary_layer)

        else:
            x = network_module(x, nfilters[i], ratio, pools_size[i], dropouts_rate[i], i,
                               pre_act=pre_act, shortcut=shortcut, binary_layer=binary_layer)

    # Reshape
    if reshape_type == 'global_avg':
        x = GlobalAveragePooling2D()(x)

    elif reshape_type == 'flatten':
        x = Flatten()(x)

    elif reshape_type == 'global_max':
        x = GlobalMaxPooling2D()(x)

    if dense_layer is None:
        x = Dense(n_classes)(x)
        x = BatchNormalization()(x)
        x = Activation('softmax')(x)

    model = Model(ip, x)

    if verbose:
        print(model.summary())

    return model


def res_conv_standard_post_csse_split_freqs(h, w, n_channels, n_classes,
                                            nfilters, pools_size, dropouts_rate, ratio, reshape_type, dense_layer,
                                            n_split_freqs, f_split_freqs, pre_act=False, shortcut='conv',
                                            verbose=False, binary_layer=False):
    ip = Input(shape=(h, w, n_channels))

    if n_split_freqs == 2:

        splits = Lambda(freq_split, arguments={'n_split_freqs': n_split_freqs,
                                               'f_split_freqs': f_split_freqs})(ip)

        x1 = splits[0]
        x2 = splits[1]

        for i in range(0, len(nfilters)):
            x1 = network_module(x1, nfilters[i], ratio, pools_size[i], dropouts_rate[i], i,
                                pre_act=pre_act, shortcut=shortcut, binary_layer=binary_layer, trident='_1')
            x2 = network_module(x2, nfilters[i], ratio, pools_size[i], dropouts_rate[i], i,
                                pre_act=pre_act, shortcut=shortcut, binary_layer=binary_layer, trident='_2')

        x = concatenate([x1, x2], axis=1)

        # Reshape
        if reshape_type == 'global_avg':
            x = GlobalAveragePooling2D()(x)

        elif reshape_type == 'flatten':
            x = Flatten()(x)

        elif reshape_type == 'global_max':
            x = GlobalMaxPooling2D()(x)

        if dense_layer is None:
            x = Dense(n_classes)(x)
            x = BatchNormalization()(x)
            x = Activation('softmax')(x)

        model = Model(ip, x)

        if verbose:
            print(model.summary())

        return model

    elif n_split_freqs == 3:

        splits = Lambda(freq_split, arguments={'n_split_freqs': n_split_freqs,
                                               'f_split_freqs': f_split_freqs})(ip)

        x1 = splits[0]
        x2 = splits[1]
        x3 = splits[2]

        for i in range(0, len(nfilters)):
            x1 = network_module(x1, nfilters[i], ratio, pools_size[i], dropouts_rate[i], i,
                                pre_act=pre_act, shortcut=shortcut, binary_layer=binary_layer, trident='_1')
            x2 = network_module(x2, nfilters[i], ratio, pools_size[i], dropouts_rate[i], i,
                                pre_act=pre_act, shortcut=shortcut, binary_layer=binary_layer, trident='_2')
            x3 = network_module(x3, nfilters[i], ratio, pools_size[i], dropouts_rate[i], i,
                                pre_act=pre_act, shortcut=shortcut, binary_layer=binary_layer, trident='_3')

        x = concatenate([x1, x2, x3], axis=1)

        # Reshape
        if reshape_type == 'global_avg':
            x = GlobalAveragePooling2D()(x)
        elif reshape_type == 'flatten':
            x = Flatten()(x)
        elif reshape_type == 'global_max':
            x = GlobalMaxPooling2D()(x)

        if dense_layer is None:
            x = Dense(n_classes)(x)
            x = BatchNormalization()(x)
            x = Activation('softmax')(x)

        model = Model(ip, x)

        if verbose:
            print(model.summary())

        return model
