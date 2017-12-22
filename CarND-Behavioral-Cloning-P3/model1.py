# https://github.com/naokishibuya/car-behavioral-cloning.git
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE, batch_generator
import argparse
import os

np.random.seed(0)


def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv'))

    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(args):
    """
    Modified NVIDIA model
    "valid" padding means no padding
    number of filters is a hyper parameters

    160 x 320 x 3                                               66 x 200 x 3
    lambda normalization                                        lambda normalization
    160 x 320 x 3                                               66 x 200 x 3
    Conv2d (24 filters, kernel:5x5, strides:2x2)                Conv2d (24 filters, kernel:5x5, strides:2x2)
    78 x 158 x 24                                               31 x 98 x 24
    Conv2d (36 filters, kernel:5x5, strides:2x2)                Conv2d (36 filters, kernel:5x5, strides:2x2)
    37 x 77 x 36                                                14 x 47 x 36
    Conv2d (48 filters, kernel:5x5, strides:2x2)                Conv2d (48 filters, kernel:5x5, strides:2x2)
    17 x 37 x 48                                                5  x 22 x 48
    Conv2d (64 filters, kernel:3x3, strides:1x1)                Conv2d (64 filters, kernel:3x3, strides:1x1)
    15 x 35 x 64                                                3  x 20 x 64
    Conv2d (64 filters, kernel:3x3, strides:1x1)                Conv2d (64 filters, kernel:3x3, strides:1x1)
    13 x 33 x 64                                                1  x 18 x 64
    Dropout()                                                   Dropout()
    13 x 33 x 64                                                1  x 18 x 64
    Flatten                                                     Flatten
    27456                                                       1152
    Dense(100)                                                  Dense(100)
    100                                                         100
    Dense(50)                                                   Dense(50)
    50                                                          50
    Dense(10)                                                   Dense(10)
    10                                                          10
    Dense(1)                                                    Dense(1)
    1                                                           1
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation='elu', padding="valid", strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', padding="valid", strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', padding="valid", strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu', padding="valid"))
    model.add(Conv2D(64, (3, 3), activation='elu', padding="valid"))

    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    print("Total training   samples: {}".format(len(X_train)))
    print("Total validation samples: {}".format(len(X_valid)))
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')

    checkpoint = ModelCheckpoint('model_nVidia-{epoch:03d}-{val_loss:.4f}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))
    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        steps_per_epoch=len(X_train)/args.batch_size,
                        epochs=args.nb_epoch, verbose=1, callbacks=[checkpoint, earlystopping],
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        validation_steps=len(X_valid)/args.batch_size)


def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    DATA_DIR0 = '/data/udacity_data'  # Sample data from Udacity (8037 lines)
    DATA_DIR1 = '/data/testtrack_train1'  # Short smooth driving
    DATA_DIR2 = '/data/testtrack_train2'  # Short smooth driving
    DATA_DIR3 = '/data/testtrack_train3'  # With recovery driving
    DATA_DIR4 = '/data/testtrack_train4_5rounds'  # Long smooth driving (5rounds, 6390)
    DATA_DIR5 = '/data/challengetrack_train1'  # Short Challenge Track
    DATA_DIR6 = '/data/testrack_train5_10rounds'  # more training on test track
    DATA_DIR7 = '/data/testtrack_train_all'

    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default=DATA_DIR7)
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=150)
    # parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=4000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=128)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    model = build_model(args)
    train_model(model, args, *data)


if __name__ == '__main__':
    main()
