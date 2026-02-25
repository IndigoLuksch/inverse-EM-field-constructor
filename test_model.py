"""
bash: source .venv-gpu/bin/activate && python test_model.py
--> in same env as model was trained in (tensorflow 2.15)
"""
import tensorflow as tf
import numpy as np
from data import Dataset
import config
import model as Model  #for custom loss


#load model
print("\n\n---Loading model and test dataset---")
model_name = "model4.keras"
model = tf.keras.models.load_model(
    f"models/{model_name}",
    custom_objects={'custom_loss': Model.custom_loss_polar}
)
print("Model loaded")

#load test dataset
dataset = Dataset()
test_ds = tf.data.Dataset.load(dataset.local_path + "/test_ds")
test_ds = test_ds.map(dataset.normalise_data, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(config.TRAINING_CONFIG['batch_size']).prefetch(tf.data.AUTOTUNE)
print("Dataset loaded")

#calculate test steps
test_steps = int(config.DATASET_CONFIG['dataset_size'] * config.DATASET_CONFIG['test_split']) // \
             config.TRAINING_CONFIG['batch_size']

#get predictions and actual values for each data point
print("\n\n---Calculating outputs---")
predictions = []
actual = []

for i, (inputs, labels) in enumerate(test_ds.take(test_steps)):
    prediction = model.predict(inputs, verbose=0)
    predictions.append(prediction)
    actual.append(labels.numpy())
    if (i + 1) % 10 == 0:
        print(f"{i + 1}/{test_steps} batches")

#concatenate batches
predictions = np.concatenate(predictions, axis=0)
actual = np.concatenate(actual, axis=0)

#calcualte metrics
print("\n\n---Results---")
output_names = ['x position', 'y position', 'dimension a', 'dimension b', 'magnetisation magnitude', 'magnetisation direction']
output_ranges = [2*config.AOI_CONFIG['x_dim'],
                 2*config.AOI_CONFIG['y_dim'],
                 config.MAGNET_CONFIG['dim_max'] - config.MAGNET_CONFIG['dim_min'],
                 config.MAGNET_CONFIG['dim_max'] - config.MAGNET_CONFIG['dim_min'],
                 config.MAGNET_CONFIG['M_max'] - config.MAGNET_CONFIG['M_min'],
                 2 * np.pi]

for i, name in enumerate(output_names):
    mae = np.mean(np.abs(predictions[:, i] - actual[:, i]))
    mae_pc = 100 * mae / output_ranges[i]
    print(f"{name} - MAE: {mae_pc:.6f}% ")

print(f"{'='*60}")

