import config
from callbacks import lr_on_plateau, early_stopping, GetLRAfterEpoch


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


def check_callbacks():
    # TODO: optimize if else
    if config.early_stopping is not True and config.get_lr_after_epoch is not True and config.factor_lr_on_plateau is not True:
        return None
    else:
        if config.early_stopping is True:
            if config.monitor_es is None:
                monitor_es = 'val_categorical_accuracy'
            else:
                monitor_es = config.monitor_es

            if config.min_delta_es is None:
                min_delta_es = 0.00001
            else:
                min_delta_es = config.min_delta_es

            if config.mode_es is None:
                mode_es = 'auto'
            else:
                mode_es = config.mode_es

            if config.patience_es is None:
                patience_es = 50
            else:
                patience_es = config.patience_es

            es = early_stopping(monitor_es, min_delta_es, mode_es, patience_es)

        else:
            es = []

        if config.get_lr_after_epoch is True:
            get_lr = GetLRAfterEpoch()
        else:
            get_lr = []

        if config.lr_on_plateau is True:
            if config.monitor_lr_on_plateau is None:
                monitor_lr = 'val_categorical_accuracy'
            else:
                monitor_lr = config.monitor_lr_on_plateau

            if config.factor_lr_on_plateau is None:
                factor_lr = 0.5
            else:
                factor_lr = config.factor_lr_on_plateau

            if config.patience_lr_on_plateau is None:
                patience_lr = 20
            else:
                patience_lr = config.patience_lr_on_plateau

            if config.min_lr_on_plateau is None:
                min_lr = 0.000001
            else:
                min_lr = config.min_lr_on_plateau

            lr_onplt = lr_on_plateau(monitor_lr, factor_lr, patience_lr, min_lr)

        else:
            lr_onplt = []

    callbacks = [es, get_lr, lr_onplt]
    callbacks = list(filter(None, callbacks))

    return callbacks
