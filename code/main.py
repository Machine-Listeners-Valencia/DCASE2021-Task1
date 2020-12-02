import config
from load_data import load_h5s
from models import Res_3L32_CSSE
from focal_loss import categorical_focal_loss

X, Y, val_x, val_y = load_h5s(config.path)

print('Training shape: {}'.format(X.shape))
print('Validation shape: {}'.format(val_x.shape))

model = Res_3L32_CSSE(X.shape[1], X.shape[2], X.shape[3], Y.shape[1])

model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25, .25, .25, .25, .25, .25, .25, .25]],
                                           gamma=config.fl_gamma)],
              metrics=['categorical_accuracy'], optimizer='adam')

if config.quick_test:
    epochs = 2
else:
    epochs = config.epochs

model.fit(X, Y, validation_data=(val_x, val_y), batch_size=32, epochs=epochs)
