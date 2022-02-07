from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy
from os import walk

# smaller size, faster training time
img_re_size = (32, 32)
# learning in batches, bigger number faster learning speed
batch_size = 64

if __name__ == '__main__':
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    training_set = train_datagen.flow_from_directory('./dataSet/training_set',
                                                     target_size=img_re_size,
                                                     batch_size=batch_size,
                                                     class_mode='binary')

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_set = test_datagen.flow_from_directory('./dataSet/test_set',
                                                target_size=img_re_size,
                                                batch_size=batch_size,
                                                class_mode='binary')

    # initialize NN with hidden layers
    cnn = keras.Sequential(
        [
            layers.Conv2D(activation="relu", filters=32, kernel_size=3, input_shape=[img_re_size[0], img_re_size[1], 3]),
            layers.MaxPool2D(pool_size=2, strides=2),
            layers.Conv2D(activation="relu", filters=32, kernel_size=3),
            layers.MaxPool2D(pool_size=2, strides=2),
            layers.Conv2D(activation="relu", filters=32, kernel_size=3),
            layers.Flatten(),
            layers.Dense(units=128, activation="relu"),
            layers.Dense(units=2, activation="sigmoid")
        ]
    )
        #sparse_categorical_crossentropy
    cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn.fit(x=training_set, validation_data=test_set, epochs=10)
    cnn.save('./saved_model/cats_dogs')

    right_answers = 0
    wrong_answers = 0

    for (dirpath, dirnames, filenames) in walk('./dataSet/single_prediction'):
        for i in range(len(filenames)):
            print(filenames[i])
            test_image = image.load_img(path='./dataSet/single_prediction/' + filenames[i], target_size=img_re_size)
            test_image = image.img_to_array(test_image)
            test_image = numpy.expand_dims(test_image, axis=0)

            result = cnn.predict(test_image/255.0, batch_size=None)
            print('Cat: ' + "{:.0%}".format(result[0][0]))
            print('Dog: ' + "{:.0%}".format(result[0][1]))

            if result[0][0] > 0.5:
                answer = 'cat'
            elif result[0][1] > 0.5:
                answer = 'dog'
            else:
                answer = 'cant understand'

            if answer == filenames[i][0: 3]:
                right_answers += 1
            else:
                wrong_answers += 1
            print('-------------------------')
        print('Right answers: ' + str(right_answers))
        print('Wrong answers: ' + str(wrong_answers))
        print('Successes rate: ' + str(100 / (wrong_answers + right_answers) * right_answers))

