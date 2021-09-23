import os
import glob

import cv2
import tqdm
import numpy as np

from scipy.ndimage.interpolation import rotate


def horizontal_flip(image):

    image = image[:, ::-1, :]
    return image


def vertical_flip(image):

    image = image[::-1, :, :]
    return image


def rotate90(image, angle):

    image = rotate(image, angle)
    return image


def augment():

    angles = [90, 180, 270]

    train_images = glob.glob('C:/Users/coco/Downloads/new_dataset/train/**/*.png')

    # horizontal flip
    for image in tqdm.tqdm(train_images):
        output_image_path = os.path.splitext(image)[0] + '_horizon.png'
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = horizontal_flip(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_image_path, image)

    # vertical flip
    for image in tqdm.tqdm(train_images):
        output_image_path = os.path.splitext(image)[0] + '_vertical.png'
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = vertical_flip(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_image_path, image)


    # rotate90
    for angle in angles:
        for image in tqdm.tqdm(train_images):
            output_image_path = os.path.splitext(image)[0] + f'_rotate_{angle}.png'
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = rotate90(image, angle)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_image_path, image)


if __name__ == '__main__':

    # generate augmented image flip(horizontal, vertical), rotate(90, 180, 270)
    augment()