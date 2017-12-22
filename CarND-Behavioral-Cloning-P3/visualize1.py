from utils import visualize_steering_distribution, overlay_steering, load_image, random_flip, random_translate, random_shadow, random_brightness, augment, preprocess
from model1 import load_data, batch_generator
import numpy as np
import random
import argparse
import cv2

np.random.seed(0)


def main():
    """
    Visualize training/validation samples distribution
    """
    datadir = '/data/testtrack_train_all'
    parser = argparse.ArgumentParser(description='Behavioral Cloning Helping Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,     default=datadir)
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float,   default=0.2)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,     default=1000)
    args = parser.parse_args()

    X_train, X_valid, y_train, y_valid = load_data(args)

    # images, steers = batch_generator(args.data_dir, X_train, y_train, 1000, True)
    # https://stackoverflow.com/questions/1073396/is-generator-next-visible-in-python-3-0
    # data_generator = batch_generator(args.data_dir, X_train, y_train, args.batch_size, True)
    # images, steers = data_generator.__next__()

    images, steers = batch_generator(args.data_dir, X_train, y_train, args.batch_size, True)


    """
    visualize_steering_distribution(steers)

    center = "center_2016_12_01_13_38_04_108.jpg"
    left = "left_2016_12_01_13_38_04_108.jpg"
    right = "right_2016_12_01_13_38_04_108.jpg"
    image, steering_angle = augment("/data/testtrack_train_all/IMG", center, left, right, 0)
    cv2.imshow('augmented', image)
    cv2.moveWindow('augmented', 20, 20)

    image = preprocess(image)
    cv2.imshow('preprocessed', image)
    cv2.moveWindow('preprocessed', 400, 400)

    image1 = load_image("/data/testtrack_train_all/IMG", "center_2016_12_01_13_38_04_108.jpg")
    cv2.imshow('Original', image1)

    cv2.imshow('test', images[3])
    """
    """
    image = load_image("/data/testtrack_train_all/IMG", "center_2016_12_01_13_38_04_108.jpg")
    steering_angle = 0

    cv2.imshow('Original', image)
    image, steering_angle = random_flip(image, steering_angle)
    cv2.imshow('Flip', image)
    cv2.moveWindow('Flip', 20, 20)
    image, steering_angle = random_translate(image, steering_angle, 100, 10)
    cv2.imshow('translate', image)
    image = random_shadow(image)
    cv2.imshow('shadow', image)
    image = random_brightness(image)
    cv2.imshow('brightness', image)
    """
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # show image
    for i in range(0):
        # idx = random.randrange(len(images))
        idx = i
        img_with_angle = overlay_steering(images[idx], float(steers[idx]), None)
        # img_with_angle = display_image(image1, -0.3, None)
        cv2.imshow('Image with Angle', img_with_angle)
        cv2.moveWindow('Image with Angle', 20, 20)
        print (images[idx])
        # cv2.imshow('trimmed', trim_image(img))
        # cv2.moveWindow('trimmed', 70, 600)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
