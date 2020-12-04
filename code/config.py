# data paths
data_path = '/repos/DCASE2021-Task1/data/gammatone_64/'
code_path = '/repos/DCASE2021-Task1/code/'

# audio representation hyperparameters
freq_bands = 64

# model parameters
verbose = True  # [True, False]
n_filters = [32, 64, 128]
pools_size = [(1, 10), (1, 5), (1, 5)]
dropouts_rate = [0.3, 0.3, 0.3]

ratio = 2

reshape_method = 'global_avg'  # ['global_avg', 'global_max', 'flatten']
dense_layer = None
dropouts_rate_cl = None

split_freqs = False  # [True, False]
n_split_freqs = 3
f_split_freqs = [64, 128]

# callbacks
# Early Stopping
early_stopping = True  # [True, False]
monitor_es = None
min_delta_es = None
mode_es = None
patience_es = None

# Get Learning Rate
get_lr_after_epoch = True  # [True, False]

# Reduce LR OnPlateau
lr_on_plateau = True  # [True, False]
monitor_lr_on_plateau = None
factor_lr_on_plateau = None
patience_lr_on_plateau = None
min_lr_on_plateau = None

# Save models and csvs
save_outputs = True  # [True, False]
outputs_path = '/repos/DCASE2021-Task1/outputs/'
best_model_name = 'best.h5'
last_model_name = 'last.h5'
log_name = 'log.csv'

# training hyperparamteres
quick_test = True  # [True, False]
loss_type = 'focal_loss'  # ['focal_loss', 'categorical_loss']
fl_alpha = 0.25  # needed if focal
fl_gamma = 2  # needed if focal
data_augmentation = 'mixup'
epochs = 500
mixup_alpha = 0.4
batch_size = 32
training_verbose = None  # [None, 0, 1, 2]
