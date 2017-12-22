import numpy as np
import pandas as pd
import os
import matplotlib.image as mpimg
import cv2
import argparse
import random
from utils2 import load_image, resize, crop, rgb2yuv, preprocess, overlay_steering, STRAIGHT_STEERING
from utils2 import random_flip, random_translate, random_shadow, random_brightness

def typeinfo(img):
    print ("type:", end="")
    print (type(img))

    print ("shape:", end="")
    print (img.shape)

    print ("dtype", end="")
    print (img.dtype)


def main():
    """
    Random test program
    """
    data_dir = '/data/testtrack_train1'
    parser = argparse.ArgumentParser(description='Behavioral Cloning Analysis Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default=data_dir)
    args = parser.parse_args()

    steering_orig = 0.3583844
    img_orig = load_image(data_dir, 'IMG/center_2017_11_10_22_39_37_716.jpg')
    typeinfo(img_orig)

    # Visualize breakdown of augmented steps: random_flip, random_translate, random_shadow, random_brightness
    img_flip, steering_flip = random_flip(img_orig, steering_orig)
    img_trans, steering_trans = random_translate(img_flip, steering_flip, 100, 10)
    img_shadow = random_shadow(img_trans)
    steering_shadow = steering_trans
    img_brightness = random_brightness(img_shadow)
    steering_bright = steering_shadow

    all_imgs = np.vstack((np.hstack((img_flip, img_trans)), np.hstack((img_shadow, img_brightness))))

    # all_imgs = np.hstack((temp_imgs, img_brightness))
    cv2.imshow('flip, translate, shadow, brightness', all_imgs)

    # Visualize preprocessing steps: original, cropped, resize and rgb2yuv
    img_cropped = crop(img_orig)
    typeinfo(img_cropped)

    img_resize = resize(img_cropped)
    typeinfo(img_resize)

    img_yuv = rgb2yuv(img_resize)
    typeinfo(img_yuv)

    cv2.imshow('original', img_orig)
    cv2.imshow('cropped', img_cropped)
    cv2.imshow('resized', img_resize)
    cv2.imshow('yuv',       img_yuv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Visualize steering angle (ground truth and predicated) over image
    data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv'))

    centers  = data_df['center'].values
    lefts    = data_df['left'].values
    rights   = data_df['right'].values
    steers   = data_df['steering'].values

    for i in range(200):
        if abs(steers[i] < 0.5):
            continue

        center_angle = steers[i]
        left_angle  = center_angle + 0.25
        right_angle = center_angle - 0.25

        center_img  = overlay_steering(preprocess(load_image(data_dir, centers[i]), False), center_angle)
        left_img    = overlay_steering(preprocess(load_image(data_dir, lefts[i]), False), left_angle)
        right_img   = overlay_steering(preprocess(load_image(data_dir, rights[i]), False), right_angle, 0.3)

        all_imgs = np.vstack((np.vstack((left_img, center_img)), right_img))
        cv2.imshow('left({:.2f}), center({:.2f}), right({:.2f})'.format(left_angle, center_angle, right_angle), all_imgs)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
   main()
