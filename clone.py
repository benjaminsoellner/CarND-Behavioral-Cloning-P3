import csv
import cv2
import numpy as np
import sklearn
import sklearn.utils
import sys
import pandas as pd
import os.path

H5_FILE = "model_{0}.h5"
CSV_FILE = "model_{0}.csv"
MODEL = (sys.argv[1] if len(sys.argv)>0 else "simple")
AUG_FACTOR = 6


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                for source_path_column, measurement_correction in zip([0, 1, 2], [1.0, -0.2, 0.2]):
                    source_path = batch_sample[source_path_column]
                    filename = source_path.split("\/")[-1]
                    current_path = 'data/IMG/' + filename
                    image = cv2.imread(current_path)
                    images.append(image)
                    measurement = float(batch_sample[3])
                    measurements.append(measurement + measurement_correction)
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement*-1.0)
            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers import Flatten, Dense, Lambda, Cropping2D, MaxPooling2D, Dropout

model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

history = None

if MODEL == "nvidia":
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(train_generator, validation_data=validation_generator,
                        samples_per_epoch=len(train_samples)*AUG_FACTOR, nb_val_samples=len(validation_samples)*AUG_FACTOR,
                        nb_epoch=5)
elif MODEL == "nvidia_mod":
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(50))
    model.add(Dense(20))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(train_generator, validation_data=validation_generator,
                                  samples_per_epoch=len(train_samples) * AUG_FACTOR,
                                  nb_val_samples=len(validation_samples) * AUG_FACTOR,
                                  nb_epoch=8)
elif MODEL == "simple":
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(train_generator, validation_data=validation_generator,
                        samples_per_epoch=len(train_samples)*AUG_FACTOR, nb_val_samples=len(validation_samples)*AUG_FACTOR,
                        nb_epoch=5)
elif MODEL == "lenet":
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(train_generator, validation_data=validation_generator,
                        samples_per_epoch=len(train_samples) * AUG_FACTOR, nb_val_samples=len(validation_samples) * AUG_FACTOR,
                        nb_epoch=5)
elif MODEL == "lenet_dropout":
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dropout(0.3))
    model.add(Dense(84))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(train_generator, validation_data=validation_generator,
                        samples_per_epoch=len(train_samples) * AUG_FACTOR, nb_val_samples=len(validation_samples) * AUG_FACTOR,
                        nb_epoch=5)
elif MODEL == "nvidia_dropout":
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dropout(0.3))
    model.add(Dense(50))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(train_generator, validation_data=validation_generator,
                        samples_per_epoch=len(train_samples)*AUG_FACTOR, nb_val_samples=len(validation_samples)*AUG_FACTOR,
                        nb_epoch=5)

model.save(H5_FILE.format(MODEL))
history_dict = {'loss': history.history['loss'], 'val_loss': history.history['val_loss']}
pd.DataFrame(data=history_dict).to_csv(CSV_FILE.format(MODEL), index_label='epoch')