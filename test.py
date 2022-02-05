from tensorflow import keras
from keras import layers
import tensorflow
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy
from os import walk


cats_dogs_model = keras.models.load_model('./saved_model/cats_dogs')

# Check its architecture
cats_dogs_model.summary()

for (dirpath, dirnames, filenames) in walk('./dataSet/single_prediction'):
    for i in range(len(filenames)):
        print(filenames[i])
        test_image = image.load_img(path='./dataSet/single_prediction/' + filenames[i], target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = numpy.expand_dims(test_image, axis=0)

        result = cats_dogs_model.predict(test_image)

        if result[0][0] == 1:
            print('dog')
        else:
            print('cat')