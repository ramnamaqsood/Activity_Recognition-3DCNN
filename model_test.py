import tensorflow
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.losses import binary_crossentropy
import numpy
import os
import numpy as np
import cv2 as cv
import argparse

############################### ARGUMENT PARSER ###############################
#parser = argparse.ArgumentParser(description='Use this script to run action recognition using 3D ResNet34',
#                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('--input', '-i', help='Path to input video file. Skip this argument to capture frames from a camera.')
#parser.add_argument('--model', required=True, help='Path to model.')
#parser.add_argument('--classes', default=findFile('action_recongnition_kinetics.txt'), help='Path to classes list.')

# Loading JSON Model from the disk:
json_file = open('./Saved_Model/ucf_json_model_exp_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
print ('[INFO] >> Model Loaded JSON File from the Disk')

# Populate Weights into the JSON Model:
model.load_weights('./Saved_Model/ucf_hd5_weights_exp_1.h5')
print ('[INFO] >> Model Loaded Successfully from the Disk')

# Model Summary:
model.summary()

# Model Compilation:
le = 0.001
opt = SGD(lr=le, momentum=0.009, decay=le)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
print ('[INFO] >> Model Compiled Successfully !\n\n')

#print('********************** Model Weights (Format 1)**********************')
#model_weights = model.get_weights()
#print(model_weights)

print('********************** Model Configuration **********************')
total_layers = 0
for layer in model.layers:
    layer_weights = layer.get_weights()
    total_layers += 1
    print('Layer Name:',layer.name)
    print('Layer Configuration:',layer.get_config())
    print('Layer Input Shape:', layer.input_shape)
    print('Layer Output Shape:', layer.output_shape)
    #print('Layer Weights',layer_weights)
print('Total Layers in Network: ', total_layers)