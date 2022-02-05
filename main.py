from tensorflow import keras
from keras import layers
import tensorflow
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy
from os import walk

if __name__ == '__main__':
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    training_set = train_datagen.flow_from_directory('./dataSet/training_set',
                                                     target_size=(64, 64),
                                                     batch_size=32,
                                                     class_mode='binary')

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_set = test_datagen.flow_from_directory('./dataSet/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

    # initialize NN with hidden layers
    cnn = keras.Sequential(
        [
            layers.Conv2D(activation="relu", filters=32, kernel_size=3, input_shape=[64, 64, 3]),
            layers.MaxPool2D(pool_size=2, strides=2),
            layers.Conv2D(activation="relu", filters=32, kernel_size=3),
            layers.MaxPool2D(pool_size=2, strides=2),
            layers.Flatten(),
            layers.Dense(units=128, activation="relu"),
            layers.Dense(units=1, activation="sigmoid")
        ]
    )

    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn.fit(x=training_set, validation_data=test_set, epochs=25)
    cnn.save('./saved_model/cats_dogs')

    for (dirpath, dirnames, filenames) in walk('./dataSet/single_prediction'):
        for i in range(len(filenames)):
            print(filenames[i])
            test_image = image.load_img(path='./dataSet/single_prediction/'+filenames[i], target_size=(64, 64))
            test_image = image.img_to_array(test_image)
            test_image = numpy.expand_dims(test_image, axis=0)

            result = cnn.predict(test_image)

            print(training_set.class_indices)

            if result[0][0] == 1:
                print('dog')
            else:
                print('cat')
