#STEP 1: First we parse the arguments
import sys
import argparse

parser = argparse.ArgumentParser(description='Model training')
parser.add_argument(
    'training_dir',
    type=str,
    help='path to training directory e.g. recordings/driving_iram_1.'
)
parser.add_argument(
    'model',
    type=int,
    help='The model type: 1 (Basic), 2 (LeNet), 3 (Nvidia)'
)
parser.add_argument(
    'flip',
    type=int,
    nargs='?',
    default=0,
    help='1(flip image to augment data), 0(default, normal image)'
)
parser.add_argument(
    'reuse',
    type=int,
    nargs='?',
    default=1,
    help='1(Reuse the model), 0(Create new model). Default is 1.'
)

args = parser.parse_args()
model_file=''
if args.model == 1:
    model_file='basic.h5'
elif args.model == 2:
    model_file='lenet.h5'
else:
    model_file='nvidia.h5'
    
reuse=True
if args.reuse == 0:
    reuse=False
    
training_dir=args.training_dir
if training_dir[-1] != '/':
    training_dir += '/'

flip=False
if args.flip == 1:
    flip=True
    
print('------------------------------------')
print('training data: ', training_dir, '\nmodel file: ', model_file, '\nreuse model file(if present): ', reuse, '\nflip images: ', flip)
print('------------------------------------')

proceed=input('Please review the above input arguments, Proceed with training (y/n)?')
if proceed != 'y':
    print('Exiting program.')
    sys.exit()


#STEP 2: load images
import csv
import cv2
import numpy as np

samples=[]
def load_from_dir(local_path):
    with open(local_path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

load_from_dir(training_dir)
print('no. of images: ', len(samples))

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print('training samples: ', len(train_samples))
print('validation samples: ', len(validation_samples))

aws_gpu_path = training_dir + 'IMG/'

import sklearn

def get_image_and_meas(image_path, measurement):
    name = aws_gpu_path + image_path.split('/')[-1]
    image = cv2.imread(name)
    angle = measurement
    if flip:
        image = np.fliplr(image)
        angle = -angle
    return image, angle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            correction = 0.2
            for batch_sample in batch_samples:
                measurement = float(batch_sample[3])
                center_image, center_angle = get_image_and_meas(batch_sample[0], measurement)
                left_image, left_angle = get_image_and_meas(batch_sample[1], measurement + correction)
                right_image, right_angle = get_image_and_meas(batch_sample[2], measurement - correction)
                images.extend((center_image, left_image, right_image))
                angles.extend((center_angle, left_angle, right_angle))

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#Now the model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, MaxPooling2D
#from keras import backend as ktf

def createBasicModel():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160, 320, 3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def createLeNetModel():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160, 320, 3)))

    #output of crop 90x320
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    
    #first cnn layers
    #output of this layer 88x318x6
    model.add(Conv2D(
        # new versions of keras have better way of giving input means 6, (3,3)
        # the keras documentation is for the latest version (slightly diff from here)
        6, 3, 3,
        border_mode='valid',
        activation='relu',
))

    #output 44x159x6
    model.add(MaxPooling2D())

    #2nd conv layer 
    #output 40x155x16 
    model.add(Conv2D(
        16, 5, 5,
        border_mode='valid',
        activation='relu',
))

    #output 20x77x16
    #model.add(MaxPooling2D())
    
    # output 24640 
    model.add(Flatten())
    
    # fc 1 output 300
    # this will need huge(st) no. of params
    model.add(Dense(300))

    #bringing it down to single output
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model


def createNvidiaModel():
    model = Sequential()
    #resize images in a lambda layer 
    #courtesy http://stackoverflow.com/questions/42260265/resizing-an-input-image-in-a-keras-lambda-layer
    #model.add(Lambda(lambda img: ktf.resize_images(img, 160/160, 320/320, 'tf'),
    #                 input_shape=(160, 320, 3)))

    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))

    #Since resize is problematic, using MaxPooling2D should give same effect
    model.add(MaxPooling2D())

    #output of crop 90x320
    model.add(Cropping2D(cropping=((25,10), (0,0))))
    

    #1st cnn layer
    #output of this layer 43x158x24
    model.add(Conv2D(
        24, 5, 5, 
        subsample=(2,2),
        border_mode='valid'
))

    #2nd cnn layer
    #output of this layer 20x77x36
    model.add(Conv2D(
        36, 5, 5, 
        #subsample=(2,2),
        border_mode='valid'
))

    #3rd cnn layer
    #output of this layer 8x37x48
    model.add(Conv2D(
        48, 5, 5, 
        subsample=(2,2),
        border_mode='valid'
))

    #4th cnn layer
    #output of this layer 6x35x64
    model.add(Conv2D(
        64, 3, 3, 
        border_mode='valid'
))

    #5th cnn layer
    #output of this layer 4x33x64
    model.add(Conv2D(
        64, 3, 3, 
        border_mode='valid'
))

    #6th cnn layer
    #output of this layer 2x31x64
    #model.add(Conv2D(
    #    64, 3, 3, 
   #     border_mode='valid'
#))

    # output: 3968
    model.add(Flatten())
    
    #fc1 
    model.add(Dense(100))

    #fc2 
    model.add(Dense(50))

    #fc3 
    model.add(Dense(10))

    #single output (which maps to steering o/p)
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model


def load_model_from_file():
    if reuse:
        try:
            model = load_model(model_file)
            return model
        except:
            print('Error in loading model from file:', model_file)

    my_model=None
    if args.model == 1:
        my_model = createBasicModel()
    if args.model == 2:
        my_model = createLeNetModel()
    if args.model == 3:
        my_model = createNvidiaModel()
    
    my_model.summary()
    return my_model

model = load_model_from_file()
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples)*3, validation_data=validation_generator, \
            nb_val_samples=len(validation_samples)*3, nb_epoch=3)
model.save(model_file)
