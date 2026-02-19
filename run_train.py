#import libraries
import numpy as np
import tensorflow as tf

#import python modules
import data
import config
import model as Model

#---configure GPU for Apple Silicon---
print('---Configuring GPU---')
# List available devices
print("Available devices:")
for device in tf.config.list_physical_devices():
    print(f"  {device}")

#-----------
#---VIBE----
# Get GPU devices
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid taking all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU acceleration enabled: {len(gpus)} GPU(s) found")
        print(f"  Device: {gpus[0]}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("⚠ No GPU found - running on CPU")
print('Complete\n')
#---END VIBE---
#--------------

#---generate data and save to gcloud---
print('Generating data')
generator = data.Dataset()
generator.local_path = './data/12-02-26'
generator.generate_cuboid_data_TF_dataset()

'''
#---load datasets---
print('---Loading datasets---')
train_ds = tf.data.Dataset.load("./data/12-02-26/train_ds")
val_ds = tf.data.Dataset.load("./data/12-02-26/val_ds")
print('Complete\n\n')

#---create and train model---
print('---Creating model---')
model = Model.create_model()
print('Complete\n\n')

print('---Training model---')

history = Model.train_model(model, train_dataset, val_dataset, initial_lr=config.TRAINING_CONFIG['initial_lr'], prop_to_load=prop_to_load)
print('Complete\n\n')

print("Script complete")
'''