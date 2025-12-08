import os

#paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

#create directories if they do not exist already
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

'''
NOTES 
• all SI units
'''

MAGNET_CONFIG = {
    'height' : 0.1, #m
    'dim_min' : 0.1, #m
    'dim_max' : 5, #m
    'M_min' : 0.2, #T
    'M_max' : 1.48, #T
}

#area of interest -- area containing desired magnetic field
AOI_CONFIG = {
    'x_dim' : 30, #m
    'y_dim' : 30, #m
    'resolution' : 0.1 #m
}

'''
VIBE
'''

#model parameters (paper Section II.D)
MODEL_CONFIG = {
    'name': 'ResNeXt-50',
    'input_shape': (224, 224, 2),  #H_x and H_y
    'output_dim': 5,  #x, y, a, Mx, My
    'cardinality': 32,  #number of groups
    'base_width': 4,  #channels per group
}

#training parameters (paper Section II.D)
TRAINING_CONFIG = {
    'dataset_size': 60000, #change to 60 000
    'train_split': 0.6,
    'val_split': 0.3,
    'test_split': 0.1,
    'batch_size': 60,
    'epochs': 100,
    'initial_lr': 0.1,
    'lr_decay_factor': 0.1,
    'lr_decay_epochs': [30, 60],  #decay lr at these epochs
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'loss': 'mse',
}

#data normalisation ranges
'''
NORMALIZATION = {
    'x': (MAGNET_CONFIG['position_x_min'], MAGNET_CONFIG['position_x_max']),
    'y': (MAGNET_CONFIG['position_y_min'], MAGNET_CONFIG['position_y_max']),
    'a': (MAGNET_CONFIG['side_length_min'], MAGNET_CONFIG['side_length_max']),
    'Mx': (-1.0, 1.0),  #(|M| - 0.5) * cos(phi)
    'My': (-1.0, 1.0),  #(|M| - 0.5) * sin(phi)
}
'''

RANDOM_SEED = 42
