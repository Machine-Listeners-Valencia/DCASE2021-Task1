# data paths
path = '/repos/DCASE2021-Task1/data/gammatone_64/'

# audio representation hyperparameters
freq_bands = 64

# model parameters
verbose = True
n_filters = [32, 64, 128]
pools_size = [(1, 10), (1, 5), (1, 5)]
dropouts_rate = [0.3, 0.3, 0.3]

ratio = 2

reshape_method = 'global_avg'
dense_layer = None
dropouts_rate_cl = None

# callbacks
# Early Stopping
early_stopping = True
monitor_es = None
min_delta_es = None
mode_es = None
patience_es = None

# Get Learning Rate
get_lr_after_epoch = True

# Reduce LR OnPlateau
lr_on_plateau = True
monitor_lr_on_plateau = None
factor_lr_on_plateau = None
patience_lr_on_plateau = None
min_lr_on_plateau = None

# training hyperparamteres
quick_test = True
loss_type = 'focal_loss'
fl_alpha = 0.25  # needed if focal
fl_gamma = 2  # needed if focal
data_augmentation = 'mixup'
epochs = 500
mixup_alpha = 0.4
batch_size = 32
training_verbose = None
