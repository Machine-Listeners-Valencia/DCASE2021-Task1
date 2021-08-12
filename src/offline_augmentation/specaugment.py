import os
from specAugment import spec_augment_tensorflow
import h5py
import numpy as np
from tqdm import tqdm
import tensorflow as tf


home = os.getenv('HOME')

path2data = home + '/repos/DCASE2021-Task1/data/gammatone_64/' + 'train_val_gammatone_mono_f1.h5'

hf = h5py.File(path2data, 'r')
features = hf['features'][:]
labels = hf['labels'][:]
hf.close()

#spec = features[0]
#spec = np.squeeze(spec, axis=2)

warped_masked_spectrogram = np.zeros(features.shape)

for i in tqdm(range(0, features.shape[0])):
    tf.keras.backend.clear_session()
    warped_masked_spectrogram[i] = spec_augment_tensorflow.spec_augment(mel_spectrogram=features[i],
                                                                        frequency_masking_para=int(features[i].shape[0]/4),
                                                                        time_masking_para=int(features[i].shape[1]/5))

f = h5py.File(home + '/repos/DCASE2021-Task1/data/gammatone_64/' + 'train_val_gammatone_mono_f1_specaug.h5', 'w')

f.create_dataset('features', warped_masked_spectrogram)
f.create_dataset('labels', labels)

f.close()