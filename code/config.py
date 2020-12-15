# data paths
data_path: str = '/repos/DCASE2021-Task1/data/gammatone_64/'
code_path: str = '/repos/DCASE2021-Task1/code/'

# audio representation hyperparameters
# freq_bands = 64

# model parameters
verbose: bool = True  # [True, False]
n_filters: list = [32, 64, 128]
pools_size: list = [(1, 10), (1, 5), (1, 5)]
dropouts_rate: list = [0.3, 0.3, 0.3]

ratio: int = 2
pre_act: bool = False
shortcut: str = 'global_avg'  # ['conv', 'global_avg', 'global_max']

reshape_method: str = 'global_avg'  # ['global_avg', 'global_max', 'flatten']
dense_layer = None
dropouts_rate_cl = None

split_freqs: bool = False  # [True, False]
n_split_freqs = 3
f_split_freqs = [64, 128]

# callbacks
# Early Stopping
early_stopping: bool = True  # [True, False]
monitor_es = None
min_delta_es = None
mode_es = None
patience_es = None

# Get Learning Rate
get_lr_after_epoch: bool = True  # [True, False]

# Reduce LR OnPlateau
lr_on_plateau: bool = True  # [True, False]
monitor_lr_on_plateau = None
factor_lr_on_plateau = None
patience_lr_on_plateau = None
min_lr_on_plateau = None

# Save models and csvs
save_outputs: bool = True  # [True, False]
outputs_path: str = '/repos/DCASE2021-Task1/outputs/'
best_model_name: str = 'best.h5'
last_model_name: str = 'last.h5'
log_name: str = 'log.csv'

# training hyperparamteres
quick_test: bool = True  # [True, False]
loss_type: str = 'focal_loss'  # ['focal_loss', 'categorical_loss']
fl_alpha: float = 0.25  # needed if focal
fl_gamma: int = 2  # needed if focal
data_augmentation: str = 'mixup'
epochs: int = 500
mixup_alpha: float = 0.4
batch_size: int = 32
training_verbose = None  # [None, 0, 1, 2]
