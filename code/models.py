import keras.layers
from modules import channel_spatial_squeeze_excite

from keras.models import Model

def Res_3L32_CSSE(H, W, n_channels, n_classes, verbose=False): # TODO: relative number of filters in config, as well as dropouts, maxpools
    """
    Model
    """

    ip = keras.layers.Input(shape=(H, W, n_channels))

    x1 = ip

    x = keras.layers.Conv2D(32, 3, padding='same')(ip)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU()(x)

    x = keras.layers.Conv2D(32, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x1 = keras.layers.Conv2D(32, 1, padding='same')(x1)
    x1 = keras.layers.BatchNormalization()(x1)

    x = keras.layers.add([x, x1])
    x = keras.layers.ELU()(x)

    x = channel_spatial_squeeze_excite(x, ratio=2)

    # -------------------------------------------
    # OPTIONAL
    x = keras.layers.add([x, x1])
    # x = ELU()(x)
    # -------------------------------------------

    # x = MaxPooling2D(pool_size=(2,10))(x)
    x = keras.layers.MaxPooling2D(pool_size=(1, 10))(x)
    # x = MaxPooling2D(pool_size=(1,2))(x)
    x = keras.layers.Dropout(0.3)(x)

    # ------------------------------------------------------------------------------

    x2 = x

    x = keras.layers.Conv2D(64, 3, padding='same')(x)
    # x = SeparableConv2D(64, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU()(x)
    # x = squeeze_excite_block(x, ratio=8)

    x = keras.layers.Conv2D(64, 3, padding='same')(x)
    # x = SeparableConv2D(64, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x2 = keras.layers.Conv2D(64, 1, padding='same')(x2)
    x2 = keras.layers.BatchNormalization()(x2)

    x = keras.layers.add([x, x2])
    x = keras.layers.ELU()(x)

    x = channel_spatial_squeeze_excite(x, ratio=2)
    # x = spatial_squeeze_excite_block(x)

    # -------------------------------------------
    # OPTIONAL
    x = keras.layers.add([x, x2])
    # x = ELU()(x)
    # -------------------------------------------

    x = keras.layers.MaxPooling2D(pool_size=(1, 5))(x)
    # x = MaxPooling2D(pool_size=(1,2))(x)
    x = keras.layers.Dropout(0.3)(x)

    # ------------------------------------------------------------------------------

    x3 = x

    x = keras.layers.Conv2D(128, 3, padding='same')(x)
    # x = SeparableConv2D(128, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU()(x)

    x = keras.layers.Conv2D(128, 3, padding='same')(x)
    # x = SeparableConv2D(128, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x3 = keras.layers.Conv2D(128, 1, padding='same')(x3)
    x3 = keras.layers.BatchNormalization()(x3)

    x = keras.layers.add([x, x3])
    x = keras.layers.ELU()(x)

    x = channel_spatial_squeeze_excite(x, ratio=2)
    # x = spatial_squeeze_excite_block(x)

    # -------------------------------------------
    # OPTIONAL
    x = keras.layers.add([x, x3])
    # x = ELU()(x)
    # -------------------------------------------

    x = keras.layers.MaxPooling2D(pool_size=(1, 5))(x)
    # x = MaxPooling2D(pool_size=(1,2))(x)
    x = keras.layers.Dropout(0.3)(x)

    # x = Flatten()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    # x = Dense(100)(x)
    # x = BatchNormalization()(x)
    # x = ELU()(x)

    # x = Dropout(0.4)(x)

    x = keras.layers.Dense(n_classes)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('softmax')(x)

    model = Model(ip, x)

    if verbose:

        print(model.summary())

    return model
