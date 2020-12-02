# data paths
path = '/repos/DCASE2021-Task1/data/gammatone_64/'

# audio representation hyperparameters
freq_bands = 64

# callbacks


# training hyperparamteres
quick_test = True
loss_type = 'focal_loss'
fl_alpha = 0.25 # needed if focal
fl_gamma = 2 # needed if focal
data_augmentation = 'mixup'
epochs = 500
mixup_alpha = 0.4
batch_size = 32