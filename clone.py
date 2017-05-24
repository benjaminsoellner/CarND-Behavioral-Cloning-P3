import csv
import cv2
import numpy as np
import sklearn
import sklearn.utils
import pandas as pd
from keras import backend as K

H5_FILE = "model_{0}.h5"
CSV_FILE = "model_{0}.csv"
AUG_FACTOR = 3


def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                for source_path_column, measurement_correction in zip([0, 1, 2], [0.0, -0.1, 0.1]):
                    source_path = batch_sample[source_path_column]
                    filename = source_path.split("/")[-1]
                    current_path = 'data/IMG/' + filename
                    image = cv2.imread(current_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    measurement = float(batch_sample[3])
                    measurements.append(measurement + measurement_correction)
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                if np.random.uniform()>0.5:
                    augmented_images.append(image)
                    augmented_measurements.append(measurement)
                else:
                    augmented_images.append(cv2.flip(image, 1))
                    augmented_measurements.append(measurement*-1.0)
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    skipline = True
    for line in reader:
        if skipline == True:
            skipline = False
        else:
            samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, ELU

model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

# # let model pick best channel from linear combination RGB
# model.add(Convolution2D(1,1,1,activation="relu"))

# # NVidia pipeline
# model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
# model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
# model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
# model.add(Convolution2D(64,3,3,activation="relu"))
# model.add(Convolution2D(64,3,3,activation="relu"))
# model.add(Flatten())
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='adam')
# history = model.fit_generator(train_generator, validation_data=validation_generator,
#                     samples_per_epoch=len(train_samples)*AUG_FACTOR, nb_val_samples=len(validation_samples)*AUG_FACTOR,
#                     nb_epoch=5)


# Comma.ai model
model.add(Convolution2D(16,8,8,subsample=(4,4),border_mode="same",activation="elu"))
model.add(Convolution2D(32,5,5,subsample=(2,2),border_mode="same",activation="elu"))
model.add(Convolution2D(64,5,5,subsample=(2,2),border_mode="same",activation="elu"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator, validation_data=validation_generator,
                    samples_per_epoch=len(train_samples)*AUG_FACTOR, nb_val_samples=len(validation_samples)*AUG_FACTOR,
                    nb_epoch=5)

model.save(H5_FILE.format("1"))
history_dict = {'loss': history.history['loss'], 'val_loss': history.history['val_loss']}
pd.DataFrame(data=history_dict).to_csv(CSV_FILE.format("1"), index_label='epoch')

# del K._SESSION