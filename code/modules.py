from keras.layers import (GlobalAveragePooling2D, GlobalMaxPooling2D, Dense,
                          multiply, add, Permute, Conv2D,
                          Reshape, BatchNormalization, ELU, MaxPooling2D, Dropout, Lambda)
import keras.backend as K
import warnings
import numpy as np


def _obtain_input_shape(input_shape,
                        default_size,
                        min_size,
                        data_format,
                        require_flatten,
                        weights=None):
    """Internal utility to compute/validate a model's tensor shape.
    # Arguments
        input_shape: Either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_size: Default input width/height for the model.
        min_size: Minimum input width/height accepted by the model.
        data_format: Image data format to use.
        require_flatten: Whether the model is expected to
            be linked to a classifier via a Flatten layer.
        weights: One of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).
            If weights='imagenet' input channels must be equal to 3.
    # Returns
        An integer shape tuple (may include None entries).
    # Raises
        ValueError: In case of invalid argument values.
    """
    if weights != 'imagenet' and input_shape and len(input_shape) == 3:
        if data_format == 'channels_first':
            if input_shape[0] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with {input_shape}'
                    ' input channels.'.format(input_shape=input_shape[0]))
            default_shape = (input_shape[0], default_size, default_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with {n_input_channels}'
                    ' input channels.'.format(n_input_channels=input_shape[-1]))
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)
    if weights == 'imagenet' and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting `include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be {default_shape}.'.format(default_shape=default_shape))
        return default_shape
    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[0] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape={input_shape}`'.format(input_shape=input_shape))
                if ((input_shape[1] is not None and input_shape[1] < min_size) or
                        (input_shape[2] is not None and input_shape[2] < min_size)):
                    raise ValueError('Input size must be at least {min_size}x{min_size};'
                                     ' got `input_shape={input_shape}`'.format(min_size=min_size,
                                                                               input_shape=input_shape))
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[-1] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape={input_shape}`'.format(input_shape=input_shape))
                if ((input_shape[0] is not None and input_shape[0] < min_size) or
                        (input_shape[1] is not None and input_shape[1] < min_size)):
                    raise ValueError('Input size must be at least {min_size}x{min_size};'
                                     ' got `input_shape={input_shape}`'.format(min_size=min_size,
                                                                               input_shape=input_shape))
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape={input_shape}`'.format(input_shape=input_shape))
    return input_shape


def _tensor_shape(tensor):
    return getattr(tensor, '_keras_shape')


def squeeze_excite_block(input_tensor, ratio=16):
    """ Create a channel-wise squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    """
    init = input_tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = _tensor_shape(init)[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def spatial_squeeze_excite_block(input_tensor):
    """ Create a spatial squeeze-excite block
    Args:
        input_tensor: input Keras tensor
    Returns: a Keras tensor
    References
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False,
                kernel_initializer='he_normal')(input_tensor)

    x = multiply([input_tensor, se])
    return x


def channel_spatial_squeeze_excite(input_tensor, ratio=16):
    """ Create a spatial squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    cse = squeeze_excite_block(input_tensor, ratio)
    sse = spatial_squeeze_excite_block(input_tensor)

    x = add([cse, sse])
    return x


def conv_standard_post(inp, nfilters, ratio, index, pre_act=False, shortcut='conv'):
    """ Module presented in https://ieeexplore.ieee.org/abstract/document/9118879
    :param inp: input tensor
    :param nfilters: number of filter of convolutional layers
    :param ratio: parameter for squeeze-excitation module
    :param pre_act:
    :param shortcut:
    :return: tensor
    """

    x1 = inp
    bn_name = 'bn_' + str(index)
    elu_name = 'elu_' + str(index)
    conv_name = 'conv_' + str(index)

    if pre_act:

        x = BatchNormalization(name=bn_name + '_a')(inp)
        x = ELU(name=elu_name)(x)
        x = Conv2D(nfilters, 3, padding='same', name=conv_name + '_a')(x)

        x = BatchNormalization(name=bn_name + '_b')(x)
        x = Conv2D(nfilters, 3, padding='same', name=conv_name + '_b')(x)

    else:

        x = Conv2D(nfilters, 3, padding='same', name=conv_name + '_a')(inp)
        x = BatchNormalization(name=bn_name + '_a')(x)
        x = ELU(name=elu_name)(x)

        x = Conv2D(nfilters, 3, padding='same', name=conv_name + '_b')(x)
        x = BatchNormalization(name=bn_name + '_b')(x)

    if shortcut == 'conv':
        x1 = Conv2D(nfilters, 1, padding='same', name=conv_name + '_shortcut')(x1)
        x1 = BatchNormalization(name=bn_name + '_shortcut')(x1)
    else:
        x1 = Lambda(pad_matrix_global, arguments={'type': shortcut}, name='lambda_padding_' + str(index))(x1)

    if K.int_shape(x)[3] != K.int_shape(x1)[3]:
        x = add(
            [x, Lambda(lambda y: K.repeat_elements(y, rep=int(K.int_shape(x)[3] // K.int_shape(x1)[3]), axis=3),
                       name='lambda_add_' + str(index) + '_a')(x1)])
    else:
        x = add([x, x1])
    x = ELU(name=elu_name + '_after_addition')(x)

    x = channel_spatial_squeeze_excite(x, ratio=ratio)

    if K.int_shape(x)[3] != K.int_shape(x1)[3]:
        x = add(
            [x, Lambda(lambda y: K.repeat_elements(y, rep=int(K.int_shape(x)[3] // K.int_shape(x1)[3]), axis=3),
                       name='lambda_add_' + str(index) + '_b')(x1)])
    else:
        x = add([x, x1])

    return x


def network_module(inp, nfilters, ratio, pool_size, dropout_rate, index, pre_act=False, shortcut='conv'):
    """ Implementation presented in https://ieeexplore.ieee.org/abstract/document/9118879
    :param inp: input tensor
    :param nfilters: number of filter of convolutional layers
    :param ratio: parameter for squeeze-excitation module
    :param pool_size: size of the pool
    :param dropout_rate: rate for dropout
    :param index:
    :param pre_act: pre_activation flag
    :param shortcut:
    :return:
    """
    x = conv_standard_post(inp, nfilters, ratio, index, pre_act=pre_act, shortcut=shortcut)

    x = MaxPooling2D(pool_size=pool_size, name='pool_' + str(index))(x)
    x = Dropout(dropout_rate, name='dropout_' + str(index))(x)

    return x


def pad_matrix_global(inp, type='global_avg'):
    h = K.int_shape(inp)[1]
    w = K.int_shape(inp)[2]

    if type == 'global_avg':
        x1 = GlobalAveragePooling2D()(inp)
    elif type == 'global_max':
        x1 = GlobalMaxPooling2D()(inp)

    x1_rep = K.repeat(x1, h * w)
    x1_rep = Reshape((K.int_shape(x1)[1], h, w))(x1_rep)
    x1_rep = K.permute_dimensions(x1_rep, (0, 2, 3, 1))

    return x1_rep


def freq_split(inp, n_split_freqs, f_split_freqs):
    if n_split_freqs == 2:
        x1 = inp[:, 0:f_split_freqs[0], :, :]
        x2 = inp[:, f_split_freqs[0]:, :, :]

        return [x1, x2]

    if n_split_freqs == 3:
        x1 = inp[:, 0:f_split_freqs[0], :, :]
        x2 = inp[:, f_split_freqs[0]:f_split_freqs[1], :, :]
        x3 = inp[:, f_split_freqs[1]:, :, :]

        return [x1, x2, x3]
