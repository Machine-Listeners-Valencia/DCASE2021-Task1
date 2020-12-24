# Author: Mark McDonnell, mark.mcdonnell@unisa.edu.au
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputSpec, Conv2D
#from tensorflow.keras import initializers


class BinaryConv2D(Conv2D):
    '''Binarized Convolution2D layer
    References:
    "BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1" [http://arxiv.org/abs/1602.02830]

    adapated by Mark McDonnell from https://github.com/DingKe/nn_playground/blob/master/binarynet/binary_layers.py
    '''

    def __init__(self, filters, **kwargs):
        super(BinaryConv2D, self).__init__(filters, **kwargs)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        input_dim = int(input_shape[channel_axis])

        if input_dim is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        self.multiplier = np.sqrt(
            2.0 / np.float(self.kernel_size[0]) / np.float(self.kernel_size[1]) / float(input_dim))

        self.kernel = self.add_weight(shape=self.kernel_size + (input_dim, self.filters),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):

        binary_kernel = self.kernel + K.stop_gradient(K.sign(self.kernel) - self.kernel)
        binary_kernel = binary_kernel + K.stop_gradient(binary_kernel * self.multiplier - binary_kernel)

        outputs = K.conv2d(inputs,
                           binary_kernel,
                           strides=self.strides,
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate)

        return outputs

    def get_config(self):
        config = {'multiplier': self.multiplier}
        base_config = super(BinaryConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
