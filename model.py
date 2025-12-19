from tensorflow.keras import layers, models, optimizers, callbacks
import magpylib as magpy
from tensorflow.keras.losses import MeanSquaredError
from keras import backend as K
from tensorflow.keras.applications import ResNet50
import numpy as np
import tensorflow as tf

import config
import data
import magnet_field_tf

Dataset = data.Dataset()

def create_model(input_shape=config.MODEL_CONFIG['input_shape'], output_dim=config.MODEL_CONFIG['output_dim']):
    '''
    Creates a ResNet50 model using model parameters from config
    '''
    base_model = ResNet50(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling='avg' #use global average pooling rather than flattening (flattening forces model to train on oddly shaped vectors that do not capture the shape of the data)
    )

    output = layers.Dense(output_dim, activation=None)(base_model.output) #create output layer of right shape, linear activation

    model = models.Model(inputs=base_model.input, outputs=output)

    print("ResNet50 model created")
    return model

def compute_loss_single(params_true, params_pred):
    """
    computes loss for pair of parameters values
    """
    #convert to np
    params_true = np.array(params_true, dtype=np.float32)
    params_pred = np.array(params_pred, dtype=np.float32)

    #penalise negative dimensions (magpylib doesn't allow negative dimensions)
    if np.min(params_pred[2:4]) < 0:
        return 4.0 * (1.0 + np.sum(np.abs(params_pred[2:4][params_pred[2:4] < 0])))

    #create magnets
    magnet_true = magpy.magnet.Cuboid(
        position=(float(params_true[0]), float(params_true[1]), 2.5),
        dimension=(float(params_true[2]), float(params_true[3]), 1),
        polarization=(float(params_true[4]), float(params_true[5]), 0)
    )
    magnet_pred = magpy.magnet.Cuboid(
        position=(float(params_pred[0]), float(params_pred[1]), 2.5),
        dimension=(float(params_pred[2]), float(params_pred[3]), 1),
        polarization=(float(params_pred[4]), float(params_pred[5]), 0)
    )

    #compute H
    H_true = magpy.getH(magnet_true, Dataset.points)
    H_pred = magpy.getH(magnet_pred, Dataset.points)

    return np.mean((H_true - H_pred)**2).astype(np.float32)

def custom_loss(params_true, params_pred):
    """
    Pure TensorFlow implementation of magnetic field loss.

    MUCH faster than the magpylib wrapper because:
    - No Python/NumPy function calls
    - Automatic differentiation (analytical gradients)
    - GPU acceleration
    - No sequential processing
    """

    # Convert observation points to TensorFlow tensor
    observation_points = tf.constant(Dataset.points, dtype=tf.float32)

    # Add penalty for negative dimensions
    # Use soft constraint for smooth gradients
    negative_penalty = tf.reduce_sum(tf.maximum(0.0, -params_pred[:, 2:4])**2)

    # Compute physics loss using TensorFlow implementation
    physics_loss = magnet_field_tf.compute_field_mse_loss(
        params_true,
        params_pred,
        observation_points
    )

    # Total loss = physics + constraint penalty
    total_loss = physics_loss + 10.0 * negative_penalty

    return total_loss

def compile_model(model, initial_lr=0.1):
    optimizer = optimizers.SGD(
        learning_rate=initial_lr,
        momentum=config.TRAINING_CONFIG['momentum'],
        weight_decay=1e-4,
        nesterov=True
    )

    model.compile(
        optimizer=optimizer,
        loss=custom_loss,
        metrics=[custom_loss] #config.TRAINING_CONFIG['loss_metric'],
    )

    return model

def create_callbacks():
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    csv_logger = callbacks.CSVLogger('training_history.csv', append=True)

    return [early_stopping, csv_logger]

def train_model(model, train_dataset, val_dataset, initial_lr=0.1):
    """
    Train the model using tf.data.Dataset objects loaded from GCS

    Args:
        model: Keras model to train
        train_dataset: tf.data.Dataset for training (already batched and preprocessed)
        val_dataset: tf.data.Dataset for validation (already batched and preprocessed)
        initial_lr: Initial learning rate
        steps_per_epoch: Number of batches per epoch (optional)
        validation_steps: Number of validation batches (optional)
    """
    #calc steps for terminal progress bar display
    steps_per_epoch = int(config.DATASET_CONFIG['dataset_size'] * config.DATASET_CONFIG['train_split']) // \
                      config.TRAINING_CONFIG['batch_size']
    validation_steps = int(config.DATASET_CONFIG['dataset_size'] * config.DATASET_CONFIG['val_split']) // \
                       config.TRAINING_CONFIG['batch_size']

    #compile, create callbacks
    model = compile_model(model, initial_lr)
    print("Model compiled")
    callback_list = create_callbacks()
    print("Callbacks created")

    #train :)
    history = model.fit(
        train_dataset,
        epochs=config.TRAINING_CONFIG['epochs'],
        validation_data=val_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callback_list,
        verbose=1  #show progress bar for both training and validation
    )
    print("Model trained")

    # Save trained model
    model_path = f'{config.MODEL_DIR}/trained_model.keras'
    model.save(model_path)
    print(f"Model saved to {model_path}")

    #save history
    history_path = f'{config.LOG_DIR}/training_history.npz'
    np.savez(history_path, **history.history)
    print(f"Training history saved to {history_path}")

    return history

if __name__ == '__main__':
    create_model()
