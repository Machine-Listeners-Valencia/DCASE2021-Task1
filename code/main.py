import config
from load_data import load_h5s
from models import res_conv_standard_post_csse
from focal_loss import categorical_focal_loss
from utils import check_reshape_variable, check_model_depth, check_alpha_list, check_loss_type

check_reshape_variable(config.reshape_method)
check_model_depth(config.n_filters, config.pools_size, config.dropouts_rate)
check_loss_type(config.loss_type)

X, Y, val_x, val_y = load_h5s(config.path)

print('Training shape: {}'.format(X.shape))
print('Validation shape: {}'.format(val_x.shape))

model = res_conv_standard_post_csse(X.shape[1], X.shape[2], X.shape[3], Y.shape[1],
                                    config.n_filters, config.pools_size, config.dropouts_rate, config.ratio,
                                    config.reshape_method, config.dense_layer,
                                    verbose=True)

if type(config.fl_alpha) is not list:
    alpha_list = [[config.fl_alpha] * Y.shape[1]]
else:
    alpha_list = config.fl_alpha
    check_alpha_list(alpha_list, Y.shape[1])

if config.loss_type == 'focal_loss':
    model.compile(loss=[categorical_focal_loss(alpha=alpha_list, gamma=config.fl_gamma)],
                  metrics=['categorical_accuracy'], optimizer='adam')

elif config.loss_type == 'categorical_loss':
    model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')

if config.quick_test:
    epochs = 2
else:
    epochs = config.epochs

model.fit(X, Y, validation_data=(val_x, val_y), batch_size=config.batch_size, epochs=epochs)
