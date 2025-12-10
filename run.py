#import libraries
import numpy as np

#import python modules
import data
import config
import model as Model

#---generate data and save to gcloud---
print('Generating data')
generator = data.Dataset()
generator.setup_gcloud()
generator.generate_cubiod_data(samples_per_batch=1000)

#---create, train model---
model = Model.create_model()