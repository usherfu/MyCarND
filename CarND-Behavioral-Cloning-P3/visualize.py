from data_processor import *
import cv2
from sklearn.utils import shuffle as sk_shuffle

DATA_DIR0 = '/data/udacity_data/'      # Sample data from Udacity (8037 lines)
DATA_DIR1 = '/data/testtrack_train1/'  # Short smooth driving
DATA_DIR2 = '/data/testtrack_train2/'  # Short smooth driving
DATA_DIR3 = '/data/testtrack_train3/'  # With recovery driving
DATA_DIR4 = '/data/testtrack_train4_5rounds/'  # Long smooth driving (5rounds, 6390)

raw_samples = get_samples(DATA_DIR0) + get_samples(DATA_DIR1) + get_samples(DATA_DIR2) + get_samples(DATA_DIR3)+ get_samples(DATA_DIR4)
visualize_sample_distribution(raw_samples)

raw_samples = balance_samples(raw_samples)
visualize_sample_distribution(raw_samples)

# for sample in raw_samples:
# raw_samples = shuffle(raw_samples)

for i in range(10):
    idx = random.randrange(len(raw_samples))
    sample = raw_samples[idx]
#    if abs(float(sample[3])) > 0.2:
    img = cv2.imread(sample[0])
    img_withangle = display_image(img, float(sample[3]), None, 0)
    cv2.imshow('Image with Angle', img_withangle)
    cv2.moveWindow('Image with Angle', 20, 20)

#   cv2.imshow(sample[0], img)
    flip_img, angle = flip_image(img, float(sample[3]))
#   cv2.imshow('flipped', flip_img)
    rotated_img, angle = rotate_image(img, float(sample[3]))
#   cv2.imshow('rotated', flip_img)
    bright_img, angle = random_brightness(img, float(sample[3]))
#   cv2.imshow('brightness', bright_img)
    all_imgs = np.vstack((np.hstack((img, flip_img)), np.hstack((rotated_img, bright_img))))
    cv2.imshow('original,flip,rotated, bright', all_imgs)
    cv2.moveWindow('original,flip,rotated, bright', 1100, 20)

    cv2.imshow('trimmed', trim_image(img))
    cv2.moveWindow('trimmed', 70, 600)

    cv2.waitKey(0)
    cv2.destroyAllWindows()




