import dcase_util
import numpy
import dill
import pathlib
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
from focal_loss import categorical_focal_loss


def get_keras_model_size(keras_model, verbose=True, ui=None, excluded_layers=None):
    """Calculate keras model size (non-zero parameters on disk)
    Parameters
    ----------
    keras_model : keras.models.Model
        keras model for the size calculation
    verbose : bool
        Print layer by layer information
        Default value True
    ui : dcase_util.ui.FancyLogger or dcase_util.ui.FancyPrinter
        Print handler
        Default value None
    excluded_layers : list
        List of layers to be excluded from the calculation
        Default value [keras.layers.normalization.BatchNormalization, kapre.time_frequency.Melspectrogram]
    Returns
    -------
    nothing
    """

    parameters_count = 0
    parameters_count_nonzero = 0

    parameters_bytes = 0
    parameters_bytes_nonzero = 0

    if verbose and ui is None:
        # Initialize print handler
        ui = dcase_util.ui.ui.FancyPrinter()

    if excluded_layers is None:
        # Populate excluded_layers list
        excluded_layers = []
        try:
            import keras

        except ImportError:
            raise ImportError('Unable to import keras module. You can install it with `pip install keras`.')

        excluded_layers.append(
            keras.layers.normalization.BatchNormalization
        )

        # Include kapre layers only if kapre is installed
        try:
            import kapre
            excluded_layers.append(
                kapre.time_frequency.Melspectrogram
            )
        except ImportError:
            pass

    if verbose:
        # Set up printing
        ui.row_reset()
        ui.row(
            'Name', 'Param', 'NZ Param', 'Size', 'NZ Size',
            widths=[30, 12, 12, 30, 30],
            types=['str', 'int', 'int', 'str', 'str'],
            separators=[True, False, True, False]
        )
        ui.row_sep()

    for l in keras_model.layers:
        # Loop layer by layer

        current_parameters_count = 0
        current_parameters_count_nonzero = 0
        current_parameters_bytes = 0
        current_parameters_bytes_nonzero = 0

        weights = l.get_weights()
        for w in weights:
            current_parameters_count += numpy.prod(w.shape)
            current_parameters_count_nonzero += numpy.count_nonzero(w.flatten())

            if w.dtype in ['single', 'float32', 'int32', 'uint32']:
                bytes = 32 / 8

            elif w.dtype in ['float16', 'int16', 'uint16']:
                bytes = 16 / 8

            elif w.dtype in ['double', 'float64', 'int64', 'uint64']:
                bytes = 64 / 8

            elif w.dtype in ['int8', 'uint8']:
                bytes = 8 / 8

            else:
                print('UNKNOWN TYPE', w.dtype)

            current_parameters_bytes += numpy.prod(w.shape) * bytes
            current_parameters_bytes_nonzero += numpy.count_nonzero(w.flatten()) * bytes

        if l.__class__ not in excluded_layers:
            parameters_count += current_parameters_count
            parameters_count_nonzero += current_parameters_count_nonzero

            parameters_bytes += current_parameters_bytes
            parameters_bytes_nonzero += current_parameters_bytes_nonzero

        if verbose:
            ui.row(
                l.name,
                current_parameters_count,
                current_parameters_count_nonzero,
                dcase_util.utils.get_byte_string(current_parameters_bytes, show_bytes=False),
                dcase_util.utils.get_byte_string(current_parameters_bytes_nonzero, show_bytes=False),
            )

    if verbose:
        ui.row_sep()
        ui.row(
            'Total',
            parameters_count,
            parameters_count_nonzero,
            dcase_util.utils.get_byte_string(parameters_bytes, show_bytes=True),
            dcase_util.utils.get_byte_string(parameters_bytes_nonzero, show_bytes=True),
        )
        ui.line()

    return {
        'parameters': {
            'all': {
                'count': parameters_count,
                'bytes': parameters_bytes
            },
            'non_zero': {
                'count': parameters_count_nonzero,
                'bytes': parameters_bytes_nonzero
            }
        }
    }


def get_lite_model_size(interpreter, verbose=True, ui=None):
    """Calculate TFLite model size (non-zero parameters on disk)
    Parameters
    ----------
    interpreter : TFLite model
        for the size calculation
    verbose : bool
        Print layer by layer information
        Default value True
    ui : dcase_util.ui.FancyLogger or dcase_util.ui.FancyPrinter
        Print handler
        Default value None
    Returns
    -------
    nothing
    """

    parameters_count = 0
    parameters_count_nonzero = 0

    parameters_bytes = 0
    parameters_bytes_nonzero = 0

    if verbose and ui is None:
        # Initialize print handler
        ui = dcase_util.ui.ui.FancyPrinter()

    if verbose:
        # Set up printing
        ui.row_reset()
        ui.row(
            'Name', 'Param', 'NZ Param', 'Size', 'NZ Size',
            widths=[30, 12, 12, 30, 30],
            types=['str', 'int', 'int', 'str', 'str'],
            separators=[True, False, True, False]
        )
        ui.row_sep()

    layer_details = interpreter.get_tensor_details()
    interpreter.allocate_tensors()
    for l in layer_details:
        # Loop layer by layer
        current_parameters_count = 0
        current_parameters_count_nonzero = 0
        current_parameters_bytes = 0
        current_parameters_bytes_nonzero = 0
        w = interpreter.get_tensor(l['index'])

        # Calculate only the quantize layers
        if l['dtype'] == numpy.float16:

            current_parameters_count += numpy.prod(w.shape)
            current_parameters_count_nonzero += numpy.count_nonzero(w.flatten())

            if w.dtype in ['single', 'float32', 'int32', 'uint32']:
                bytes = 32 / 8

            elif w.dtype in ['float16', 'int16', 'uint16']:
                bytes = 16 / 8

            elif w.dtype in ['double', 'float64', 'int64', 'uint64']:
                bytes = 64 / 8

            elif w.dtype in ['int8', 'uint8']:
                bytes = 8 / 8

            else:
                print('UNKNOWN TYPE', w.dtype)

            current_parameters_bytes += numpy.prod(w.shape) * bytes
            current_parameters_bytes_nonzero += numpy.count_nonzero(w.flatten()) * bytes
            # Add to total
            parameters_count += current_parameters_count
            parameters_count_nonzero += current_parameters_count_nonzero

            parameters_bytes += current_parameters_bytes
            parameters_bytes_nonzero += current_parameters_bytes_nonzero

            if verbose:
                ui.row(
                    l['name'].split('/')[-1],
                    current_parameters_count,
                    current_parameters_count_nonzero,
                    dcase_util.utils.get_byte_string(current_parameters_bytes, show_bytes=False),
                    dcase_util.utils.get_byte_string(current_parameters_bytes_nonzero, show_bytes=False),
                )

    if verbose:
        ui.row_sep()
        ui.row(
            'Total',
            parameters_count,
            parameters_count_nonzero,
            dcase_util.utils.get_byte_string(parameters_bytes, show_bytes=True),
            dcase_util.utils.get_byte_string(parameters_bytes_nonzero, show_bytes=True),
        )
        ui.line()

    return {
        'parameters': {
            'all': {
                'count': parameters_count,
                'bytes': parameters_bytes
            },
            'non_zero': {
                'count': parameters_count_nonzero,
                'bytes': parameters_bytes_nonzero
            }
        }
    }


def get_tflite(path2model):
    model = load_model(path2model, custom_objects={'categorical_focal_loss_fixed': dill.loads(
        dill.dumps(categorical_focal_loss(gamma=2., alpha=[[.25, .25, .25, .25, .25, .25, .25, .25, .25, 0.25]])))})

    filename, file_extension = os.path.splitext(path2model)

    model2convert = filename + '_tflite.h5'
    model.save(model2convert, include_optimizer=False)

    model_for_tflite = load_model(model2convert)
    # converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model2convert)
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_tflite)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    # print(type(converter))
    tfm = converter.convert()

    tfmodel_file = pathlib.Path(filename + '.tflite')
    tfmodel_file.write_bytes(tfm)

    return filename + '.tflite'


def get_tlife_size(tflite_model):
    interpreter = tf.lite.Interpreter(model_path=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_data = numpy.array(numpy.zeros(input_shape), dtype=numpy.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    get_lite_model_size(interpreter, verbose=True)


if __name__ == '__main__':
    tf_lite_path = '../outputs/2021-03-02-20-53-best/best.tflite'

    keras_model = load_model('../outputs/2021-03-02-20-53-best/best.h5', custom_objects={'categorical_focal_loss_fixed': dill.loads(
        dill.dumps(categorical_focal_loss(gamma=2., alpha=[[.25, .25, .25, .25, .25, .25, .25, .25, .25, 0.25]])))})
    print(keras_model.summary())

    get_keras_model_size(keras_model)

    get_tlife_size(tf_lite_path)
