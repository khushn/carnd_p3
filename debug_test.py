import argparse

parser = argparse.ArgumentParser(description='Model debug')
parser.add_argument(
    'model',
    type=str,
    help='The model file name e.g. lenet.h5'
)

args = parser.parse_args()

import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D

model = load_model(args.model)


debug_path='./recordings/debug/'
samples=[]
def load_from_dir(local_path):
    with open(local_path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

load_from_dir(debug_path)
print('no. of images: ', len(samples))
print('try')

for line in samples:
    name = debug_path + 'IMG/' + line[0].split('/')[-1]
    print('testing image: ', name)
    image = cv2.imread(name)
    image_array = np.asarray(image)
    print('shape of image array:', image_array.shape)

    steering_angle = float(model.predict(image_array[None,:, :, :], batch_size=1))
    print('steering angle:', steering_angle)


