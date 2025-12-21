# Inverse EM field constructor

Goal: 
- construct an arrangement of magnets that produces a desired magnetic field

Approach: 
1. train a neural network to predict properties of a single magnet given the magnetic field it produces
2. input a desired magnetic field into the model
3. subtract the magnetic field produced by the magnet it outputs
4. repeat this cycle

## Method 

Uses a ResNet50 model (not pretrained) 

Different parameter combinations may produce quite similar fields and parameters MSE loss is quite sensitive, so custom hybrid loss (linear combination of MSE of predicted parameters and MSE of H field resulting from predicted parameters) is used 

## Results 

A ResNet50 model was trained on 60000 data points (cuboidal magnets in a 2D plane). 

![val loss](scratch/val_loss.png "Training: validation loss")

*Results: regular MSE loss:*

- x position - MAE: 0.043976% 
- y position - MAE: 0.043822% 
- d imension a - MAE: 3.798878% 
- dimension b - MAE: 3.903161% 
- Mx magnetization - MAE: 13.210454% 
- My magnetization - MAE: 13.171558%
- _mean: 5.695%_

*Results: hybrid loss*
- x position - MAE: 0.020873% 
- y position - MAE: 0.019991% 
- dimension a - MAE: 3.004341% 
- dimension b - MAE: 3.097148% 
- Mx magnetization - MAE: 11.849303% 
- My magnetization - MAE: 12.219321%
- _mean: 5.035%_

(The script was containerised using Docker and ready to run in Google Cloud, but GPUs were unavailable so I ended up running it locally but storing the dataset in Google Cloud.)

## Files 

[data.py](data.py), [model.py](model.py), [config.py](config.py) are the core files required to create model, generate data, use Google Cloud. 

[run_train.py](run_train.py) is used to train the model. [run_inv.py](run_inv.py) is used to predict magnet parameters using residual error correction. 

[Dockerfile](Dockerfile), [docker-compose.yaml](docker-compose.yaml) are needed to containerise this process. (I made some notes [here](scratch/docker-gcloud-notes).)

[E_field_notebook](1_initial_attempt/E_field_notebook.ipynb) notebook contains initial attempts create a convolutional NN for an electric field

## List of models
Model files are large so I have not uploaded them here. Contact me if you would like access :) 

model1: params MSE loss, field calculated in plane of magnet 

model2: params MSE loss, field calculated just above magnet --> less extreme values 

model3: hybrid loss (field : params = 1 : 20) 
