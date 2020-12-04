import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
import math
import keras
from keras import backend as K


def plot_loss_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


# plot acc history
def plot_acc_history(history):
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


# Ex: sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
def decay_rate_x_epoch(learning_rate, epochs):  # time-based learning rate schedule

    decay_rate = learning_rate / epochs

    return decay_rate


# reduce learning rate after specific epochs
def lr_specific_epoch(lr_init, lr_decay, spec_epoch):
    def step_decay(epoch):
        if epoch % spec_epoch == 0:
            factor = int(epoch / spec_epoch)
            lr = lr_init * (lr_decay ** factor)
        else:
            factor = int(epoch / spec_epoch)
            if factor == 0:
                lr = lr_init
            else:
                lr = lr_init * (lr_decay ** factor)
        return float(lr)

    return LearningRateScheduler(step_decay)


# x = decay_rate_drop_based(0.01, 0.9, 20)
def decay_rate_drop_based(lr_init, drop, epochs_drop):  # drop learning rate at specific times during training
    def step_decay(epoch):
        lr_rate = lr_init * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lr_rate

    return LearningRateScheduler(step_decay)


def lr_on_plateau(monitor, factor, patience, min_lr):
    lr = keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=factor,
                                           patience=patience, min_lr=min_lr)
    return lr


def early_stopping(monitor, min_delta, mode, patience):
    es = keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, min_delta=min_delta,
                                       patience=patience)
    return es


# print learning rate after epoch
class GetLRAfterEpoch(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = K.eval(self.model.optimizer.lr)
        print('LR: {:.6f}'.format(lr))


class DelayedEarlyStopping(EarlyStopping):
    def on_epoch_end(self, epoch, logs=None):
        if epoch >= 100:
            super().on_epoch_end(epoch, logs=logs)





# EOF
