def check_reshape_variable(reshape_method):

    possible_options = ['global_avg', 'global_max', 'flatten']

    if reshape_method not in possible_options:
        raise Exception('Defined reshape method not available. Please check config file. '
                        'Possible options are: {}'.format(possible_options))


def check_model_depth(n_filters, pools_size, dropouts_rate):

    if len(n_filters) != len(pools_size) or len(n_filters) != len(dropouts_rate) or len(pools_size) != len(dropouts_rate):
        raise Exception('Lengths of list denoting the number of filters, dropouts rate and pool sizes do not match, '
                        'please check config file')
