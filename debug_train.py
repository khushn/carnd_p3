import csv
import cv2
import numpy as np

training_dir='./recordings/debug/'
model_file='basic.h5'
flip=False

samples=[]
def load_from_dir(local_path):
    with open(local_path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

load_from_dir(training_dir)
print('no. of images: ', len(samples))

aws_gpu_path = training_dir + 'IMG/'


def get_image_and_meas(image_path, measurement):
    name = aws_gpu_path + image_path.split('/')[-1]
    image = cv2.imread(name)
    angle = measurement
    if flip:
        image = np.fliplr(image)
        angle = -angle
    return image, angle


images = []
angles = []
correction = 0.2
for batch_sample in samples:
    measurement = float(batch_sample[3])
    center_image, center_angle = get_image_and_meas(batch_sample[0], measurement)
    images.append(center_image)
    angles.append(center_angle)
    #left_image, left_angle = get_image_and_meas(batch_sample[1], measurement + correction)
    #right_image, right_angle = get_image_and_meas(batch_sample[2], measurement - correction)
    #images.extend((center_image, left_image, right_image))
    #angles.extend((center_angle, left_angle, right_angle))

# trim image to only see section with road
X_train = np.array(images)
y_train = np.array(angles)


#Now the model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D

def createBasicModel():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160, 320, 3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

model = createBasicModel()
model.fit(X_train, y_train, validation_split=0.0, nb_epoch=1)
model.save(model_file)
