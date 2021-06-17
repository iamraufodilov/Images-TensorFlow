# load libraries
import numpy as np
import time

import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
#import tensorflow_hub as hub


# load model from hub
IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
])


# load dataset
img_path = 'G:/rauf/STEPBYSTEP/Tutorial/TensorFlow/Images TensorFlow/Tf Hub/shark.jpg'
img_shark = Image.open(img_path).resize(IMAGE_SHAPE)
#_>img_shark.show()

img_shark = np.array(img_shark)/255.0
#_>print(img_shark.shape)


# predict the image
result = classifier.predict(img_shark[np.newaxis, ...])
#_>print(result.shape)

# to get class id 
predicted_class = np.argmax(result[0], axis=-1)
#_>print(predicted_class)


