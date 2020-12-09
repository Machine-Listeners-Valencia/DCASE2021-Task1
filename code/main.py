import numpy as np

import config
from data_augmentation import MixupGenerator
from focal_loss import categorical_focal_loss
from load_data import load_h5s
from models import res_conv_standard_post_csse, res_conv_standard_post_csse_split_freqs
from tests import (check_reshape_variable, check_model_depth, check_alpha_list, check_loss_type, check_data_generator,
                   check_training_verbose, is_boolean, check_callbacks)

# check config options
check_reshape_variable(config.reshape_method)
check_model_depth(config.n_filters, config.pools_size, config.dropouts_rate)
check_loss_type(config.loss_type)
check_data_generator(config.data_augmentation)
tr_verbose = check_training_verbose(config.training_verbose)
is_boolean(config.quick_test)

# loading training data
X, Y, val_x, val_y = load_h5s(config.data_path)

print('Training shape: {}'.format(X.shape))
print('Validation shape: {}'.format(val_x.shape))

# creating model

if config.split_freqs is not True:
    model = res_conv_standard_post_csse(X.shape[1], X.shape[2], X.shape[3], Y.shape[1],
                                        config.n_filters, config.pools_size, config.dropouts_rate, config.ratio,
                                        config.reshape_method, config.dense_layer,
                                        pre_act=config.pre_act, verbose=config.verbose)

else:
    model = res_conv_standard_post_csse_split_freqs(X.shape[1], X.shape[2], X.shape[3], Y.shape[1],
                                                    config.n_filters, config.pools_size, config.dropouts_rate,
                                                    config.ratio,
                                                    config.reshape_method, config.dense_layer,
                                                    config.n_split_freqs, config.f_split_freqs,
                                                    pre_act=config.pre_act, verbose=config.verbose)

# checking focal loss if necessary
if config.loss_type == 'focal_loss':
    if type(config.fl_alpha) is not list:
        alpha_list = [[config.fl_alpha] * Y.shape[1]]
    else:
        alpha_list = config.fl_alpha
        check_alpha_list(alpha_list, Y.shape[1])

# compiling model
if config.loss_type == 'focal_loss':
    model.compile(loss=[categorical_focal_loss(alpha=alpha_list, gamma=config.fl_gamma)],
                  metrics=['categorical_accuracy'], optimizer='adam')

elif config.loss_type == 'categorical_loss':
    model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')

# number of epochs
if config.quick_test:
    epochs = 2
else:
    epochs = config.epochs

if config.data_augmentation == 'mixup':
    train_datagen = MixupGenerator(X, Y, batch_size=config.batch_size, alpha=config.mixup_alpha)()

callbacks = check_callbacks()

if config.data_augmentation is not None:
    history = model.fit_generator(train_datagen,
                                  validation_data=(val_x, val_y), epochs=epochs,
                                  steps_per_epoch=np.ceil((X.shape[0] - 1) / config.batch_size),
                                  callbacks=callbacks,
                                  verbose=tr_verbose)
else:
    history = model.fit(X, Y, validation_data=(val_x, val_y), batch_size=config.batch_size, epochs=epochs,
                        callbacks=callbacks,
                        verbose=tr_verbose)
