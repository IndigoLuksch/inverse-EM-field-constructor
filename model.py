import tensorflow as tf
import tensorflow.keras.applications.resnet50 as resnet50 #https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50

model = resnet50.ResNet50(weights='imagenet', include_top=False)
#print(model.summary())
#float32 inputs