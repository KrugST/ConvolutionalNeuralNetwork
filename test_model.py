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

right_answers = 0
wrong_answers = 0
img_re_size = (32, 32)

for (dirpath, dirnames, filenames) in walk('./dataSet/single_prediction/'):
    for i in range(len(filenames)):

        test_image = image.load_img(path='./dataSet/single_prediction/' + filenames[i], target_size=img_re_size)
        #test_image = image.load_img(path='./dataSet/training_set/dogs/' + filenames[i], target_size=img_re_size)
        test_image = image.img_to_array(test_image)
        test_image = numpy.expand_dims(test_image, axis=0)

        result = cats_dogs_model.predict(test_image/255.0)

        if result[0][0] > 0.5:
            answer = 'cat'
        elif result[0][1] > 0.5:
            answer = 'dog'
        else:
            answer = 'cant understand'

        if answer == filenames[i][0: 3]:
            right_answers += 1
        else:
            print(filenames[i])
            print('Cat: ' + "{:.0%}".format(result[0][0]))
            print('Dog: ' + "{:.0%}".format(result[0][1]))
            wrong_answers += 1
            print('-------------------------')
    print('Right answers: ' + str(right_answers))
    print('Wrong answers: ' + str(wrong_answers))
    print('Successes rate: ' + str(100 / (wrong_answers + right_answers) * right_answers))
