def check_reshape_variable(reshape_method):
    possible_options = ['global_avg', 'global_max', 'flatten']

    if reshape_method not in possible_options:
        raise Exception('Defined reshape method not available. Please check config file. '
                        'Possible options are: {}'.format(possible_options))


def check_model_depth(n_filters, pools_size, dropouts_rate):
    if len(n_filters) != len(pools_size) or len(n_filters) != len(dropouts_rate) or len(pools_size) != len(
            dropouts_rate):
        raise Exception('Lengths of list denoting the number of filters, dropouts rate and pool sizes do not match, '
                        'please check config file')


def check_alpha_list(alpha_list, n_classes):
    if len(alpha_list) != 1 or len(alpha_list[0]) != n_classes:
        raise Exception('Alpha dimensions do not match. Please check that it has been defined with double list notation'
                        ' [[]]. If that is the case, please check that is has the same positions '
                        'as the number of classes')


def check_loss_type(loss_type):
    possible_options = ['focal_loss', 'categorical_loss']

    if loss_type not in possible_options:
        raise Exception('Defined loss type not available. Please check config file. '
                        'Possible options are: {}'.format(possible_options))


def check_data_generator(data_augmentation):
    possible_options = ['mixup', None]

    if data_augmentation not in possible_options:
        raise Exception('Defined data generator not available. Please check config file. '
                        'Possible options are: {}'.format(possible_options))


def check_training_verbose(training_verbose):
    possible_options = [0, 1, 2, None]

    if training_verbose not in possible_options:
        raise Exception('Defined training verbose not available. Please check config file. '
                        'Possible options are: {}'.format(possible_options))

    if training_verbose is None:
        return 1
    else:
        return training_verbose


def is_boolean(inp):
    if isinstance(inp, bool) is not True:
        raise Exception('Variable {} is not a boolean variable, please check in config file'.format(inp))
