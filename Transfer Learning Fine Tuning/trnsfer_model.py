# load libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory


# define parameters
BATCH_SIZE = 32
IMG_SIZE = (160, 160)


# load dataset
train_dir = 'G:/rauf/STEPBYSTEP/Data/Cat Dog TF/cats_and_dogs_filtered/train'
validation_dir = 'G:/rauf/STEPBYSTEP/Data/Cat Dog TF/cats_and_dogs_filtered/validation'

train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)


# we do not hve test dataset so we take 20% of validation ds as test ds
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

#_>print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
#_>print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))


# prefetch the data
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


# use data augmentation to reduce overfitting
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])


# as base model we use mobilenet2 in case this model take input -1:1 but our image now 0-255 so rescale dataset
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
#or
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)


# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


# freeze convolution layers
base_model.trainable = False


#
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)


# add a classification head
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)


# add dense layer
prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)


# put all together
inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)


# compile the model
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


# train the model
initial_epochs = 5

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CONCLUSION
'''
in this experiment we load dataset
and trained it with pretrained model such as mobilenet model
then we tried fine tune mobile net model in order to improve accuracy
'''