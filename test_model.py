from tensorflow import keras
from keras.preprocessing import image
import numpy
from os import walk

cats_dogs_model = keras.models.load_model('./saved_model/cats_dogs')

# Check its architecture
cats_dogs_model.summary()

right_answers = 0
wrong_answers = 0
img_re_size = (32, 32)

# THIS WILL ONLY PRINT WRONG ANSWERS TO CONSOLE, TO UNDERSTAND WHAT IMAGES ARE NOT RECOGNIZED
for (dirpath, dirnames, filenames) in walk('./dataSet/single_prediction/'):
    for i in range(len(filenames)):

        test_image = image.load_img(path='./dataSet/single_prediction/' + filenames[i], target_size=img_re_size)
        #test_image = image.load_img(path='./dataSet/training_set/dogs/' + filenames[i], target_size=img_re_size)
        test_image = image.img_to_array(test_image)
        test_image = numpy.expand_dims(test_image, axis=0)

        result = cats_dogs_model.predict(test_image/255.0)

        if result[0][0] == max(result[0]):
            answer = 'cat'
        elif result[0][1] == max(result[0]):
            answer = 'dog'
        elif result[0][2] == max(result[0]):
            answer = 'ele'
        else:
            answer = 'none'

        if answer == filenames[i][0: 3]:
            right_answers += 1
        else:
            print(filenames[i])
            print('Cat: ' + "{:.0%}".format(result[0][0]))
            print('Dog: ' + "{:.0%}".format(result[0][1]))
            print('Elephant: ' + "{:.0%}".format(result[0][2]))
            wrong_answers += 1
            print('-------------------------')
    print('Right answers: ' + str(right_answers))
    print('Wrong answers: ' + str(wrong_answers))
    print('Successes rate: ' + str(100 / (wrong_answers + right_answers) * right_answers))
