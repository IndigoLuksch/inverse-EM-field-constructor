import config
import model
from sklearn.model_selection import train_test_split
import numpy as np

model = model.create_model()

#---split data---
idx = np.arange(config.TRAINING_CONFIG['dataset_size'])
train_idx, val_idx = train_test_split(idx, test_size=(config.TRAINING_CONFIG['val_split'] + config.TRAINING_CONFIG['test_split']), random_state=config.RANDOM_SEED)
val_idx, test_idx = train_test_split(val_idx, test_size=(config.TRAINING_CONFIG['val_split'] * config.TRAINING_CONFIG['test_split']), random_state=config.RANDOM_SEED)