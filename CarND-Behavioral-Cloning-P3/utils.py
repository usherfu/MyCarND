import cv2
import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.utils import shuffle as sk_shuffle

# Skip steering angle close to straight line
STRAIGHT_STEERING = 0.05

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
#IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 75, 320, 3
# IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    Original Image: [160, 320, 3] --> Result dimension: [75, 320, 3]
    """
    return image[60:-25, :, :]  # numpy slicing to remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image, convert2yuv=True):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    if convert2yuv:
        image = rgb2yuv(image)  # might not be needed.
    return image


def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.25
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.25
    return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
    """
    Randomly flip the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image vertically and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image

    height, width = image.shape[0:2]
    x1, y1 = width * np.random.rand(), 0
    x2, y2 = width * np.random.rand(), height
    xm, ym = np.mgrid[0:height, 0:width]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    ratio = .25+np.random.uniform()
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    """
    Generate an augmented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle


def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    """
    Generate training image give image paths and associated steering angles
    """
    #images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    #steers = np.empty(batch_size)
    while True:
        i = 0
        images = []
        steers = []
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # eliminate samples with steering close to zero
            if abs(float(steering_angle)) < STRAIGHT_STEERING:
                continue
            # augmentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augment(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center)
            # add the image and steering angle to the batch
            images.append(preprocess(image, False))
            steers.append(steering_angle)
            i += 1
            if i == batch_size:
                break

        X_train = np.array(images)
        y_train = np.array(steers)
        # print("current batch size: {}".format(len(X_train)))
        yield sk_shuffle(X_train, y_train)


def visualize_steering_distribution(steers):
    """
    Visualize sample distribution of steering_angle
    """
    angles = []
    for sample in steers:
        # skip if speed ~= 0, not representative of driving behavior
        if abs(float(sample)) < STRAIGHT_STEERING:
            continue
        angles.append(float(sample))

    print('Total Training Samples:', len(steers))
    print('Total Samples (Steering > {}) : {}'.format(STRAIGHT_STEERING, len(angles)))

    angles = np.array(angles)
    num_bins = 23
    avg_samples_per_bin = len(angles)/num_bins
    hist, bins = np.histogram(angles, num_bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
    plt.show()


def overlay_steering(image, angle, pred_angle=None):
    """
    Overlay angle and predicted angle over the image
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    # img = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    # get height and width of image
    h, w = image.shape[0:2]

    # apply text for steering angle
    cv2.putText(image, 'angle:' + str(angle), org=(2,33), fontFace=font, fontScale=.5, color=(255,255,255), thickness=1)

    # draw a line for steering angle
    cv2.line(image, (int(w/2), int(h)), (int(w/2+angle*w/4), int(h/2)), (0, 255, 0), thickness=4)

    # draw a line for predicated angle if available
    if pred_angle is not None:
        cv2.line(image, (int(w/2), int(h)), (int(w/2+pred_angle*w/4), int(h/2)), (0, 0, 255), thickness=2)

    return image
