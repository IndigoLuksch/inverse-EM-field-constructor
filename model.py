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

    output = layers.Dense(output_dim, activation='sigmoid')(base_model.output) #sigmoid to constrain to [0, 1]

    model = models.Model(inputs=base_model.input, outputs=output)

    print("ResNet50 model created")
    return model

#cartesian magnetisation
def custom_loss_cart(params_true, params_pred):
    """
    MOSTLY VIBE CODED

    hybrid loss: linear combination of H field MSE and parameter MSE
    (sigmoid activation --> dimensions always +ve --> no need for negative dimension penalty)
    """

    #---data prep---
    observation_points = tf.constant(Dataset.points, dtype=tf.float32)
    batch_size = tf.shape(params_pred)[0]
    n_points = tf.shape(observation_points)[0]

    #denormalise
    params_true_denorm = tf.stack([
        params_true[:, 0] * (2 * config.AOI_CONFIG['x_dim']) - config.AOI_CONFIG['x_dim'],  # x: 0-1 -> -30 to 30
        params_true[:, 1] * (2 * config.AOI_CONFIG['y_dim']) - config.AOI_CONFIG['y_dim'],  # y: 0-1 -> -30 to 30
        params_true[:, 2] * (config.MAGNET_CONFIG['dim_max'] - config.MAGNET_CONFIG['dim_min']) + config.MAGNET_CONFIG['dim_min'],  # a
        params_true[:, 3] * (config.MAGNET_CONFIG['dim_max'] - config.MAGNET_CONFIG['dim_min']) + config.MAGNET_CONFIG['dim_min'],  # b
        params_true[:, 4] * (config.MAGNET_CONFIG['M_max'] - config.MAGNET_CONFIG['M_min']) + config.MAGNET_CONFIG['M_min'],  # Mx
        params_true[:, 5] * (config.MAGNET_CONFIG['M_max'] - config.MAGNET_CONFIG['M_min']) + config.MAGNET_CONFIG['M_min'],  # My
    ], axis=1)

    params_pred_denorm = tf.stack([
        params_pred[:, 0] * (2 * config.AOI_CONFIG['x_dim']) - config.AOI_CONFIG['x_dim'],
        params_pred[:, 1] * (2 * config.AOI_CONFIG['y_dim']) - config.AOI_CONFIG['y_dim'],
        params_pred[:, 2] * (config.MAGNET_CONFIG['dim_max'] - config.MAGNET_CONFIG['dim_min']) + config.MAGNET_CONFIG['dim_min'],
        params_pred[:, 3] * (config.MAGNET_CONFIG['dim_max'] - config.MAGNET_CONFIG['dim_min']) + config.MAGNET_CONFIG['dim_min'],
        params_pred[:, 4] * (config.MAGNET_CONFIG['M_max'] - config.MAGNET_CONFIG['M_min']) + config.MAGNET_CONFIG['M_min'],
        params_pred[:, 5] * (config.MAGNET_CONFIG['M_max'] - config.MAGNET_CONFIG['M_min']) + config.MAGNET_CONFIG['M_min'],
    ], axis=1)

    #---calculate true H field---
    positions_true = tf.stack([
        params_true_denorm[:, 0],
        params_true_denorm[:, 1],
        tf.fill([batch_size], 2.5)
    ], axis=1)

    dimensions_true = tf.stack([
        params_true_denorm[:, 2],
        params_true_denorm[:, 3],
        tf.ones([batch_size])
    ], axis=1)

    polarizations_true = tf.stack([
        params_true_denorm[:, 4],
        params_true_denorm[:, 5],
        tf.zeros([batch_size])
    ], axis=1)

    #all obseration points
    obs_expanded = tf.tile(tf.expand_dims(observation_points, 0), [batch_size, 1, 1])
    positions_true_rep = tf.repeat(positions_true, n_points, axis=0)
    dimensions_true_rep = tf.repeat(dimensions_true, n_points, axis=0)
    polarizations_true_rep = tf.repeat(polarizations_true, n_points, axis=0)
    obs_flat = tf.reshape(obs_expanded, [-1, 3])

    #compute H true
    observers_rel_true = obs_flat - positions_true_rep
    H_true = magnet_field_tf.compute_H_field_batch(observers_rel_true, dimensions_true_rep, polarizations_true_rep)
    H_true = tf.reshape(H_true, [batch_size, n_points, 3])

    #normalise, clip
    H_true_normalized = H_true / Dataset.H_STD
    H_true_normalized = tf.clip_by_value(H_true_normalized, -100.0, 100.0)

    #---calculate predicted H field---
    positions_pred = tf.stack([
        params_pred_denorm[:, 0],
        params_pred_denorm[:, 1],
        tf.fill([batch_size], 2.5)
    ], axis=1)

    dimensions_pred = tf.stack([
        params_pred_denorm[:, 2],
        params_pred_denorm[:, 3],
        tf.ones([batch_size])
    ], axis=1)

    polarizations_pred = tf.stack([
        params_pred_denorm[:, 4],
        params_pred_denorm[:, 5],
        tf.zeros([batch_size])
    ], axis=1)

    positions_pred_rep = tf.repeat(positions_pred, n_points, axis=0)
    dimensions_pred_rep = tf.repeat(dimensions_pred, n_points, axis=0)
    polarizations_pred_rep = tf.repeat(polarizations_pred, n_points, axis=0)

    #calc H pred
    observers_rel_pred = obs_flat - positions_pred_rep
    H_pred = magnet_field_tf.compute_H_field_batch(observers_rel_pred, dimensions_pred_rep, polarizations_pred_rep)
    H_pred = tf.reshape(H_pred, [batch_size, n_points, 3])

    #normalise, clip
    H_pred_normalized = H_pred / Dataset.H_STD
    H_pred_normalized = tf.clip_by_value(H_pred_normalized, -100.0, 100.0)

    #---compute losses---
    #physics loss: MSE between H_true_normalized and H_pred_normalized
    physics_loss_per_sample = tf.reduce_mean(tf.square(H_true_normalized - H_pred_normalized), axis=[1, 2])
    physics_loss = tf.reduce_mean(physics_loss_per_sample)

    #parameter loss: MSE between (normalised) parameters
    param_mse = tf.reduce_mean(tf.square(params_true - params_pred))

    #combine for total loss
    #ratio chosen so physics and param losses contribute roughly equally
    total_loss = 0.05 * physics_loss + param_mse

    return total_loss

#polar coords magnetisation
def custom_loss_polar(params_true, params_pred, epsilon=1e-7):
    """
    MOSTLY VIBE CODED

    hybrid loss: linear combination of H field MSE and parameter MSE
    (sigmoid activation --> dimensions always +ve --> no need for negative dimension penalty)
    
    differences to custom_loss_cart: 
    - params MSE calculated using magnetisation in polar coord
    - params first converted to cartesian for H field MSE calculation 
    """

    #---polar->cartesian coordinates for magnetisation---
    params_true_polar = params_true
    params_pred_polar = params_pred


    params_true = tf.stack([params_true[0], params_true[1], params_true[2], params_true[3],
                   tf.sqrt(tf.square(params_true[4]) + tf.square(params_true[5]) + epsilon),
                   tf.math.atan2(params_true[5] + epsilon, params_true[4] + epsilon)])
    params_pred = tf.stack([params_pred[0], params_pred[1], params_pred[2], params_pred[3],
                   tf.sqrt(tf.square(params_pred[4]) + tf.square(params_pred[5] + epsilon)),
                   tf.atan2(params_pred[5] + epsilon, params_pred[4] + epsilon)])

    #---data prep---
    observation_points = tf.constant(Dataset.points, dtype=tf.float32)
    batch_size = tf.shape(params_pred)[0]
    n_points = tf.shape(observation_points)[0]

    #denormalise
    params_true_denorm = tf.stack([
        params_true[:, 0] * (2 * config.AOI_CONFIG['x_dim']) - config.AOI_CONFIG['x_dim'],  # x: 0-1 -> -30 to 30
        params_true[:, 1] * (2 * config.AOI_CONFIG['y_dim']) - config.AOI_CONFIG['y_dim'],  # y: 0-1 -> -30 to 30
        params_true[:, 2] * (config.MAGNET_CONFIG['dim_max'] - config.MAGNET_CONFIG['dim_min']) + config.MAGNET_CONFIG['dim_min'],  # a
        params_true[:, 3] * (config.MAGNET_CONFIG['dim_max'] - config.MAGNET_CONFIG['dim_min']) + config.MAGNET_CONFIG['dim_min'],  # b
        params_true[:, 4] * (config.MAGNET_CONFIG['M_max'] - config.MAGNET_CONFIG['M_min']) + config.MAGNET_CONFIG['M_min'],  # Mx
        params_true[:, 5] * (config.MAGNET_CONFIG['M_max'] - config.MAGNET_CONFIG['M_min']) + config.MAGNET_CONFIG['M_min'],  # My
    ], axis=1)

    params_pred_denorm = tf.stack([
        params_pred[:, 0] * (2 * config.AOI_CONFIG['x_dim']) - config.AOI_CONFIG['x_dim'],
        params_pred[:, 1] * (2 * config.AOI_CONFIG['y_dim']) - config.AOI_CONFIG['y_dim'],
        params_pred[:, 2] * (config.MAGNET_CONFIG['dim_max'] - config.MAGNET_CONFIG['dim_min']) + config.MAGNET_CONFIG['dim_min'],
        params_pred[:, 3] * (config.MAGNET_CONFIG['dim_max'] - config.MAGNET_CONFIG['dim_min']) + config.MAGNET_CONFIG['dim_min'],
        params_pred[:, 4] * (config.MAGNET_CONFIG['M_max'] - config.MAGNET_CONFIG['M_min']) + config.MAGNET_CONFIG['M_min'],
        params_pred[:, 5] * (config.MAGNET_CONFIG['M_max'] - config.MAGNET_CONFIG['M_min']) + config.MAGNET_CONFIG['M_min'],
    ], axis=1)

    #---calculate true H field---
    positions_true = tf.stack([
        params_true_denorm[:, 0],
        params_true_denorm[:, 1],
        tf.fill([batch_size], 2.5)
    ], axis=1)

    dimensions_true = tf.stack([
        params_true_denorm[:, 2],
        params_true_denorm[:, 3],
        tf.ones([batch_size])
    ], axis=1)

    polarizations_true = tf.stack([
        params_true_denorm[:, 4],
        params_true_denorm[:, 5],
        tf.zeros([batch_size])
    ], axis=1)

    #all obseration points
    obs_expanded = tf.tile(tf.expand_dims(observation_points, 0), [batch_size, 1, 1])
    positions_true_rep = tf.repeat(positions_true, n_points, axis=0)
    dimensions_true_rep = tf.repeat(dimensions_true, n_points, axis=0)
    polarizations_true_rep = tf.repeat(polarizations_true, n_points, axis=0)
    obs_flat = tf.reshape(obs_expanded, [-1, 3])

    #compute H true
    observers_rel_true = obs_flat - positions_true_rep
    H_true = magnet_field_tf.compute_H_field_batch(observers_rel_true, dimensions_true_rep, polarizations_true_rep)
    H_true = tf.reshape(H_true, [batch_size, n_points, 3])

    #normalise, clip
    H_true_normalized = H_true / Dataset.H_STD
    H_true_normalized = tf.clip_by_value(H_true_normalized, -100.0, 100.0)

    #---calculate predicted H field---
    positions_pred = tf.stack([
        params_pred_denorm[:, 0],
        params_pred_denorm[:, 1],
        tf.fill([batch_size], 2.5)
    ], axis=1)

    dimensions_pred = tf.stack([
        params_pred_denorm[:, 2],
        params_pred_denorm[:, 3],
        tf.ones([batch_size])
    ], axis=1)

    polarizations_pred = tf.stack([
        params_pred_denorm[:, 4],
        params_pred_denorm[:, 5],
        tf.zeros([batch_size])
    ], axis=1)

    positions_pred_rep = tf.repeat(positions_pred, n_points, axis=0)
    dimensions_pred_rep = tf.repeat(dimensions_pred, n_points, axis=0)
    polarizations_pred_rep = tf.repeat(polarizations_pred, n_points, axis=0)

    #calc H pred
    observers_rel_pred = obs_flat - positions_pred_rep
    H_pred = magnet_field_tf.compute_H_field_batch(observers_rel_pred, dimensions_pred_rep, polarizations_pred_rep)
    H_pred = tf.reshape(H_pred, [batch_size, n_points, 3])

    #normalise, clip
    H_pred_normalized = H_pred / Dataset.H_STD
    H_pred_normalized = tf.clip_by_value(H_pred_normalized, -100.0, 100.0)

    #---compute losses---
    #physics loss: MSE between H_true_normalized and H_pred_normalized
    physics_loss_per_sample = tf.reduce_mean(tf.square(H_true_normalized - H_pred_normalized), axis=[1, 2])
    physics_loss = tf.reduce_mean(physics_loss_per_sample)

    #parameter loss: MSE between (normalised) parameters
    param_mse = tf.reduce_mean(tf.square(params_true_polar - params_pred_polar))

    #combine for total loss
    #ratio chosen so physics and param losses contribute roughly equally
    total_loss = 0.05 * physics_loss + param_mse

    return total_loss

def compile_model(model, initial_lr):
    optimizer = optimizers.legacy.Adam(
        learning_rate=initial_lr,
        clipnorm=1.0  #clip gradients to stabilise
    )

    model.compile(
        optimizer=optimizer,
        loss=custom_loss_polar,
        metrics=[custom_loss_polar] #config.TRAINING_CONFIG['loss_metric'],
    )

    return model

def create_callbacks():
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    csv_logger = callbacks.CSVLogger('training_history.csv', append=True)

    #ReduceLROnPlateau --> adaptive learning rate
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    #terminate training if loss - NaN
    terminate_nan = callbacks.TerminateOnNaN()

    return [early_stopping, csv_logger, reduce_lr, terminate_nan]

def train_model(model, train_dataset, val_dataset, initial_lr=0.1, prop_to_load=1.0):
    #calc steps for terminal progress bar display, adjusted for actual data loaded
    steps_per_epoch = int(config.DATASET_CONFIG['dataset_size'] * config.DATASET_CONFIG['train_split'] * prop_to_load) // \
                       config.TRAINING_CONFIG['batch_size']
    validation_steps = int(config.DATASET_CONFIG['dataset_size'] * config.DATASET_CONFIG['val_split'] * prop_to_load) // \
                       config.TRAINING_CONFIG['batch_size']

    #compile, create callbacks
    model = compile_model(model, initial_lr)
    print("Model compiled")
    callback_list = create_callbacks()
    print("Callbacks created")

    #train :)
    history = model.fit(
        train_dataset.repeat(),
        epochs=config.TRAINING_CONFIG['epochs'],
        validation_data=val_dataset.repeat(),
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callback_list,
        verbose=1  #show progress bar for both training and validation
    )
    print("Model trained")

    #save trained model
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
