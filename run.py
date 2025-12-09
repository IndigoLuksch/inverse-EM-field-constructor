#import libraries
from sklearn.model_selection import train_test_split
import numpy as np

#import python modules
import data
import config
import model

model = model.create_model()

#---generate data and save to gcloud---
print('Generating data')
generator = data.Dataset()
generator.setup_gcloud()
generator.generate_cubiod_data(samples_per_batch=1000)

#---split data---
idx = np.arange(config.DATASET_CONFIG['dataset_size'])
train_idx, val_idx = train_test_split(idx,
                                      test_size=(config.DATASET_CONFIG['val_split'] + config.DATASET_CONFIG['test_split']),
                                      random_state=config.RANDOM_SEED)
val_idx, test_idx = train_test_split(val_idx,
                                     test_size=(config.DATASET_CONFIG['val_split'] * config.DATASET_CONFIG['test_split']),
                                     random_state=config.RANDOM_SEED)
