# https://github.com/joshchao39/Udacity-CarND-P3-Behavioral-Cloning.git
import pickle

from keras import backend
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
from keras.layers.core import Lambda
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from data_processor import *

# DATA_DIR = './large_files/sample/'  # Sample data from Udacity
# DATA_DIR3 = './large_files/train3/'  # Short smooth driving
# DATA_DIR4 = './large_files/train4/'  # Full lap smooth driving (2 minutes)
# DATA_DIR5 = './large_files/train5/'  # Recovery lap
# DATA_DIR6 = './large_files/train6/'  # Hard corner smooth driving
# DATA_DIR7 = './large_files/train7/'  # 20 minutes smooth driving
# DATA_DIR8 = './large_files/train8/'  # The split road correction
# DATA_DIR9 = './large_files/train9/'  # 25 minutes smooth driving (slightly emphasize on hard corners)

DATA_DIR0 = '/data/udacity_data/'      # Sample data from Udacity (8037 lines)
DATA_DIR1 = '/data/testtrack_train1/'  # Short smooth driving
DATA_DIR2 = '/data/testtrack_train2/'  # Short smooth driving
DATA_DIR3 = '/data/testtrack_train3/'  # With recovery driving
DATA_DIR4 = '/data/testtrack_train4_5rounds/'  # Long smooth driving (5rounds, 6390)
DATA_DIR5 = '/data/challengetrack_train1/'  # Short Challenge Track
DATA_DIR6 = '/data/testrack_train5_10rounds/'  # more training on test track
DATA_DIR7 = '/data/testtrack_train_all/'

"""Prepare training/validation data"""
# Collect data from samples
# raw_samples = get_samples(DATA_DIR4) + get_samples(DATA_DIR7) + get_samples(DATA_DIR5, True)
# raw_samples = get_samples(DATA_DIR0) + \
#              get_samples(DATA_DIR4) + \
#              get_samples(DATA_DIR3, True) + \
#              get_samples(DATA_DIR6)
raw_samples = get_samples(DATA_DIR7)

# raw_samples = shuffle(raw_samples)

# add recovery
raw_samples_with_recovery = add_recovery_samples(raw_samples)

# balance samples against steering angle distribution
# samples = balance_samples(raw_samples)
samples = balance_samples(raw_samples_with_recovery)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# build generator for training/validation
batch_size = 100
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

row, col, ch = get_trimmed_image_size()
print("Trimmed image size:", (row, col, ch))

"""Build Model"""
model = Sequential()
# Pre-process incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(row, col, ch)))

"""replacing Keras 1.x version of Convolution2D with Conv2D in 2.x
model.add(Convolution2D(24, 3, 3, border_mode='valid', activation='relu'))
"""
model.add(Conv2D(24, (3, 3), padding="valid", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))


# model.add(Convolution2D(32, 3, 3, border_mode='valid', activation='relu'))
model.add(Conv2D(32, (3, 3), padding="valid", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))


# model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
model.add(Conv2D(64, (3, 3), padding="valid", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(300, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(50, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(20, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(1))
adam = Adam(lr=1e-4)
model.compile(loss='mse', optimizer=adam)

for layer in model.layers:
    print("Layer shape:", layer.output_shape)

"""Train"""
callback1 = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto')
callback2 = ModelCheckpoint('./model.h5', monitor='val_loss', verbose=0, save_best_only=True,
                            save_weights_only=False, mode='auto', period=1)
"""  keras fit_generator version 1.x
history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                              validation_data=validation_generator,
                              nb_val_samples=len(validation_samples),
                              nb_epoch=150,
                              callbacks=[callback1, callback2])
"""
history = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size,
                              epochs=150, verbose=1, callbacks=[callback1, callback2],
                              validation_data=validation_generator, validation_steps=len(validation_samples)/batch_size,
                              workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

print("Best validation loss:", min(history.history['val_loss']))

with open('history.pickle', 'wb') as hist_out:
    pickle.dump(history.history, hist_out)

backend.clear_session()
