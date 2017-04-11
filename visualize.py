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
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, MaxPooling2D
import scipy.misc

model = load_model(args.model)

def resize(img):
    #We must import it inside the function
    #because its used from a lambda layer
    import tensorflow as tf
    return tf.image.resize_images(img, (66, 235))

#Add visualization layer
def visualizeModel(model, layer=1):
    model2 = Sequential()
    model2.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
    
    #resize images
    model2.add(Lambda(resize))

    #normalize
    model2.add(Lambda(lambda x: x/255.0 - 0.5))
    
    #add the weights of the first model to the second model
    w = model.layers[3].get_weights()
    print('conv1: shape of model weights:', w[0][:,:,:,0].shape)
    model2.add(Conv2D(
        24, 5, 5,
        subsample=(2,2),
        activation='relu',
        border_mode='valid',
        weights=w
    ))
    if layer == 0:
        return model2

    w = model.layers[4].get_weights()
    print('conv2: shape of model weights:', w[0][:,:,:,0].shape)
    model2.add(Conv2D(
        36, 5, 5,
        subsample=(2,2),
        activation='relu',
        border_mode='valid',
        weights=w
    ))
    if layer == 1:
        return model2

    w = model.layers[5].get_weights()
    print('conv3: shape of model weights:', w[0][:,:,:,0].shape)
    model2.add(Conv2D(
        48, 5, 5,
        subsample=(2,2),
        activation='relu',
        border_mode='valid',
        weights=w
    ))
    if layer == 2:
        return model2

    w = model.layers[6].get_weights()
    print('conv4: shape of model weights:', w[0][:,:,:,0].shape)
    model2.add(Conv2D(
        64, 3, 3,
        activation='relu',
        border_mode='valid',
        weights=w
    ))
    if layer == 3:
        return model2

    w = model.layers[7].get_weights()
    print('conv5: shape of model weights:', w[0][:,:,:,0].shape)
    model2.add(Conv2D(
        64, 3, 3,
        activation='relu',
        border_mode='valid',
        weights=w
    ))
    if layer == 4:
        return model2

    return model2

    

debug_path='./recordings/debug/'
samples=[]
def load_from_dir(local_path):
    with open(local_path + 'driving_log_visualize.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

load_from_dir(debug_path)
print('no. of images: ', len(samples))
print('try')


import matplotlib.pyplot as plt
#function copied from 'traffic sign classifier project
def outputFeatureMap(sub_plots_row, image_array, keras_layer, activation_min=-1, activation_max=-1, label=""):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error keras_layer is not defined it maybe having trouble accessing the variable from inside a function
    activation = keras_layer.predict(image_array[None,:, :, :], batch_size=1)
    #print(activation)
    featuremaps = activation.shape[3]
    num_figs=min(8, featuremaps)
    pref='FeatureMap'
    sub_plots_row[0].set_title(label + " :", fontsize=6)

    for featuremap in range(num_figs):
        sub_plt = sub_plots_row[featuremap]
        sub_plt.axis("off")
        if activation_min != -1 & activation_max != -1:
            sub_plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            sub_plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            sub_plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            sub_plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
            

for line in samples:
    name = debug_path + 'IMG/' + line[0].split('/')[-1]
    print('testing image: ', name)
    image = cv2.imread(name)
    scipy.misc.imsave('visualization/original.jpg', image)
    image_array = np.asarray(image)
    #print('shape of image array:', image_array.shape)

    #steering_angle = float(model.predict(image_array[None,:, :, :], batch_size=1))
    #print('steering angle:', steering_angle)
    plt.figure(figsize=(15,15))
    num_cnns=5
    f, sub_plts = plt.subplots(num_cnns, 8)
    
    for i in range(num_cnns):
        model2=visualizeModel(model, i)
        outputFeatureMap(sub_plts[i], image_array, model2, activation_min = 0., label="cnn"+str(i+1))
    
    #bbox_inches='tight' to reduce the white space
    plt.savefig('visualization/layers.png', bbox_inches='tight')


