from keras.src.callbacks import early_stopping
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.applications import ResNet50
import numpy as np

import config

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

def compile_model(model, initial_lr=0.1):
    optimizer = optimizers.SGD(
        learning_rate=initial_lr,
        momentum=config.TRAINING_CONFIG['momentum'],
        weight_decay=1e-4,
        nesterov=True
    )

    model.compile(
        optimizer=optimizer,
        loss=MeanSquaredError(),
        metrics=[config.TRAINING_CONFIG['loss_metric']],
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

def train_model(model, initial_lr=0.1):

    model = compile_model(model)
    print("Model compiled")
    callbacks = create_callbacks()
    print("Callbacks created")

    model.fit(
        X_train, y_train,
        batch_size=config.TRAINING_CONFIG['batch_size'],
        epochs=config.TRAINING_CONFIG['epochs'],
        val = [X_val, y_val],
        callbacks=callbacks
    )
    print("Model trained")


if __name__ == '__main__':
    create_model()
