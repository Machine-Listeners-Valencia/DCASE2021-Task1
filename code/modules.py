from tensorflow.keras.layers import (GlobalAveragePooling2D, GlobalMaxPooling2D, Dense,
                                     multiply, add, Permute, Conv2D,
                                     Reshape, BatchNormalization, ELU, MaxPooling2D, Dropout, Lambda)
import tensorflow.keras.backend as K
import warnings
from complexity_considerations.binary_layer import BinaryConv2D
from tensorflow.keras.regularizers import l2

__authors__ = "Javier Naranjo, Sergi Perez and Irene Martín"
__copyright__ = "Machine Listeners Valencia"
__credits__ = ["Machine Listeners Valencia"]
__license__ = "MIT License"
__version__ = "0.4.0"
__maintainer__ = "Javier Naranjo"
__email__ = "janal2@alumni.uv.es"
__status__ = "Dev"
__date__ = "2020"


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
    # return getattr(tensor, '_keras_shape')
    return getattr(tensor, '_shape_val')


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


def spatial_squeeze_excite_block(input_tensor, binary_layer=False):
    """ Create a spatial squeeze-excite block
    Args:
        input_tensor: input Keras tensor
    Returns: a Keras tensor
    References
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    :param binary_layer:
    """

    if binary_layer is True:
        se = BinaryConv2D(1, kernel_size=1, activation='sigmoid', use_bias=False,
                          kernel_initializer='he_normal')(input_tensor)
    else:
        se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False,
                    kernel_initializer='he_normal')(input_tensor)

    x = multiply([input_tensor, se])
    return x


def channel_spatial_squeeze_excite(input_tensor, ratio=16, binary_layer=False):
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
    sse = spatial_squeeze_excite_block(input_tensor, binary_layer=binary_layer)

    x = add([cse, sse])
    return x


def conv_standard_post(inp, nfilters, ratio, index, pre_act=False, shortcut='conv', binary_layer=False):
    """ Module presented in https://ieeexplore.ieee.org/abstract/document/9118879
    :param inp: input tensor
    :param nfilters: number of filter of convolutional layers
    :param ratio: parameter for squeeze-excitation module
    :param pre_act:
    :param shortcut:
    :param binary_layer:
    :return: tensor
    """

    x1 = inp
    bn_name = 'bn_' + str(index)
    elu_name = 'elu_' + str(index)
    conv_name = 'conv_' + str(index)

    if pre_act:

        x = BatchNormalization(name=bn_name + '_a')(inp)
        x = ELU(name=elu_name)(x)

        if binary_layer is True:
            x = BinaryConv2D(nfilters, kernel_size=3, use_bias=False, padding='same', name=conv_name + '_a')(x)
        else:
            x = Conv2D(nfilters, 3, padding='same', name=conv_name + '_a')(x)

        x = BatchNormalization(name=bn_name + '_b')(x)

        if binary_layer is True:
            x = BinaryConv2D(nfilters, kernel_size=3, use_bias=False, padding='same', name=conv_name + '_b')(x)
        else:
            x = Conv2D(nfilters, 3, padding='same', name=conv_name + '_b')(x)

    else:

        if binary_layer is True:
            x = BinaryConv2D(nfilters, kernel_size=3, use_bias=False, padding='same', name=conv_name + '_a')(inp)
        else:
            x = Conv2D(nfilters, 3, padding='same', name=conv_name + '_a')(inp)
        x = BatchNormalization(name=bn_name + '_a')(x)
        x = ELU(name=elu_name)(x)

        if binary_layer is True:
            x = BinaryConv2D(nfilters, kernel_size=3, use_bias=False, padding='same', name=conv_name + '_b')(x)
        else:
            x = Conv2D(nfilters, 3, padding='same', name=conv_name + '_b')(x)
        x = BatchNormalization(name=bn_name + '_b')(x)

    if shortcut == 'conv':

        if binary_layer is True:
            x1 = BinaryConv2D(nfilters, kernel_size=1, use_bias=False, padding='same', name=conv_name + '_shortcut')(x1)
        else:
            x1 = Conv2D(nfilters, 1, padding='same', name=conv_name + '_shortcut')(x1)
        x1 = BatchNormalization(name=bn_name + '_shortcut')(x1)
    elif shortcut == 'global_avg' or shortcut == 'global_max':
        x1 = Lambda(pad_matrix_global, arguments={'type': shortcut}, name='lambda_padding_' + str(index))(x1)

    x = module_addition(x, x1, index, 'a')

    x = ELU(name=elu_name + '_after_addition')(x)

    x = channel_spatial_squeeze_excite(x, ratio=ratio, binary_layer=binary_layer)

    x = module_addition(x, x1, index, 'b')

    return x


def network_module(inp, nfilters, ratio, pool_size, dropout_rate, index, pre_act=False, shortcut='conv',
                   binary_layer=False):
    """ Implementation presented in https://ieeexplore.ieee.org/abstract/document/9118879
    :param inp: input tensor
    :param nfilters: number of filter of convolutional layers
    :param ratio: parameter for squeeze-excitation module
    :param pool_size: size of the pool
    :param dropout_rate: rate for dropout
    :param index:
    :param pre_act: pre_activation flag
    :param shortcut:
    :param binary_layer:
    :return:
    """
    x = conv_standard_post(inp, nfilters, ratio, index, pre_act=pre_act, shortcut=shortcut, binary_layer=binary_layer)

    x = MaxPooling2D(pool_size=pool_size, name='pool_' + str(index))(x)
    x = Dropout(dropout_rate, name='dropout_' + str(index))(x)

    return x


def module_addition(inp1, inp2, index, suffix):
    """

    :param inp1:
    :param inp2:
    :param index:
    :param suffix:
    :return:
    """
    if K.int_shape(inp1)[3] != K.int_shape(inp2)[3]:
        x = add(
            [inp1, Lambda(lambda y: K.repeat_elements(y, rep=int(K.int_shape(inp1)[3] // K.int_shape(inp2)[3]), axis=3),
                          name='lambda_add_' + str(index) + '_' + str(suffix))(inp2)])
    else:
        x = add([inp1, inp2])

    return x


def pad_matrix_global(inp, type='global_avg'):
    """

    :param inp:
    :param type:
    :return:
    """
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
    """

    :param inp:
    :param n_split_freqs:
    :param f_split_freqs:
    :return:
    """
    if n_split_freqs == 2:
        x1 = inp[:, 0:f_split_freqs[0], :, :]
        x2 = inp[:, f_split_freqs[0]:, :, :]

        return [x1, x2]

    if n_split_freqs == 3:
        x1 = inp[:, 0:f_split_freqs[0], :, :]
        x2 = inp[:, f_split_freqs[0]:f_split_freqs[1], :, :]
        x3 = inp[:, f_split_freqs[1]:, :, :]

        return [x1, x2, x3]
