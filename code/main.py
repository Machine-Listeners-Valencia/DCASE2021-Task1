"""
Description: main file to execute submission system. Please README.md for further instructions and check packages
            versions as defined in [requirements.txt] and [tf1-15.yml].
            Execution configuration is defined using [config.py].

License: MIT License
"""

# Futures

# Generic/Built-in

# Other Libs
import numpy as np

# Owned
import config
from data_augmentation import MixupGenerator
from focal_loss import categorical_focal_loss
from load_data import load_h5s
from models import construct_model
from tests_variables import (check_reshape_variable, check_model_depth, check_alpha_list, check_loss_type,
                             check_data_generator,
                             check_training_verbose, is_boolean, check_callbacks, check_shortcut_type)

__authors__ = "Javier Naranjo, Sergi Perez and Irene Martín"
__copyright__ = "Machine Listeners Valencia"
__credits__ = ["Machine Listeners Valencia"]
__license__ = "MIT License"
__version__ = "0.6.0"
__maintainer__ = "Javier Naranjo"
__email__ = "janal2@alumni.uv.es"
__status__ = "Dev"
__date__ = "2020"

# {code}

# check config options
check_reshape_variable(config.reshape_method)
check_model_depth(config.n_filters, config.pools_size, config.dropouts_rate)
check_loss_type(config.loss_type)
check_data_generator(config.data_augmentation)
tr_verbose = check_training_verbose(config.training_verbose)
is_boolean(config.quick_test)
check_shortcut_type(config.shortcut)

# loading training data
x, y, val_x, val_y = load_h5s(config.home_path, config.data_path, config.validation_file, config.training_file)

print('Training shape: {}'.format(x.shape))
print('Validation shape: {}'.format(val_x.shape))

# creating model

model = construct_model(x, y)

# checking focal loss if necessary
if config.loss_type == 'focal_loss':
    if type(config.fl_alpha) is not list:
        alpha_list = [[config.fl_alpha] * y.shape[1]]
    else:
        alpha_list = config.fl_alpha
        check_alpha_list(alpha_list, y.shape[1])

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
    train_datagen = MixupGenerator(x, y, batch_size=config.batch_size, alpha=config.mixup_alpha)()

callbacks = check_callbacks(config.home_path)

if config.data_augmentation is not None:
    history = model.fit_generator(train_datagen,
                                  validation_data=(val_x, val_y), epochs=epochs,
                                  steps_per_epoch=np.ceil((x.shape[0] - 1) / config.batch_size),
                                  callbacks=callbacks,
                                  verbose=tr_verbose)
else:
    history = model.fit(x, y, validation_data=(val_x, val_y), batch_size=config.batch_size, epochs=epochs,
                        callbacks=callbacks,
                        verbose=tr_verbose)
