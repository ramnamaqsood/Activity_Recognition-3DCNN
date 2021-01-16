 # Number of Epochs for the experiment
epochs = 5

from model_32m import c3d_model
from gen_frame_list import gen_frame_list

from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from keras.optimizers import SGD,Adam,Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from keras.layers import Dense,Flatten, Conv3D, MaxPool3D,Input,BatchNormalization , Dropout
from keras.optimizers import Adadelta
from keras.optimizers import SGD,adam, Adagrad
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.models import load_model
from keras.models import save_model
from keras.utils import multi_gpu_model
from keras.models import model_from_json

import tensorboard as tensorboard
import random
import matplotlib
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import cv2
import os
from math import floor
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.keras.models import save_model

matplotlib.use('AGG')
#print("Entering with batch_size=32,img_size=150,120 having 4 biased classes+changed model")
#config = tf.ConfigProto(device_count={'GPU':0, 'CPU':4})

#----------------------- construct the argument parse and parse the arguments -----------------------
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=False,
	help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
	help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
	help="epoch to restart training at")
args = vars(ap.parse_args())

#-------------------------------------- Set_clip_size and set_batch_size-------------------------------------
batch_size = 16
no_frame = 16

#------------------------------- Set_path_trainTest_direstory_Images -----------------------------

train_dir  = "/home/jupyter/Drive_2/Event_Detection/ucf_frames/train/"
test_dir=  "/home/jupyter/Drive_2/Event_Detection/ucf_frames/test/"

class_list_train,train_frame_list = gen_frame_list(train_dir,True)
print ('[INFO] >>> Training Class List: ', class_list_train)
print ('[INFO] >>> Training Frames List: ', class_list_train)
class_list_test,test_frame_list = gen_frame_list(test_dir,True)
print ('[INFO] >>> Testing Class List: ', class_list_test)
print ('[INFO] >>> Testing Frames List: ', class_list_test)

#-------------------------- Generate DATA --------------------------------------------------
def generate_data(files_list, categories, batch_size):
    print ("[INFO] >>> Generating Data:")                      
    """"Replaces Keras' native ImageDataGenerator."""    
    if len(files_list) != 0:
        print("[INFO] >>> Total Files: ", len(files_list))
        cpe = 0 
        while True:
            if cpe == floor(len(files_list)/ (batch_size * no_frame)):
                cpe = 0
#             for cpe in range(floor(len(files_list)/ (batch_size * no_frame))):
            x_train = []
            y_train = []
#             print('Cycle No: ', cpe)
            c_start  = batch_size * cpe 
            c_end    = (c_start + batch_size)
#             print("C_Start:",c_start, " c_end: ", c_end)
            for b in range(c_start, c_end):
#                 print('  Frame Set: ',b)
                start = b *  no_frame
                end   = start + (no_frame)                    
                stack_of_16=[]
                for i in range(start,end):                  
#                     print('    Frame Index: ',i)

                    try:
                        image = cv2.imread(files_list[i])
                        image = cv2.resize(image,(100, 100))
                    except Exception as e:
                        print('[BROKEN IMAGE]: ',str(e))
                        print('Image File: ', files_list[i])
                        continue
                        
                    image = image / 255.
                    stack_of_16.append(image)
#                   print("Path : ", files_list[i])
#                 print("Class: ", files_list[start].split("/")[4])
#                 print("Cat Index: ",categories.index(files_list[start].split("/")[4]))
                y_train.append(categories.index(files_list[start].split("/")[7]))
              #  print("y_train",y_train)
                x_train.append(np.array(stack_of_16))
            cpe += 1
#                 print("y_train",np_utils.to_categorical(y_train,2))

#            print("x_train",np.array(x_train).shape)
#            print("y_train",np.array(y_train).shape)
#            print("Total Frames:_x_train ", len(x_train))
            yield(np.array(x_train).transpose(0,1,2,3,4), np_utils.to_categorical(y_train, 2))

#----------------------------- Deploying Model Training on Multiple n-GPUs -----------------------------
print ('\n\n\n\n*******************************************************************************************')
print ('ALERT: Deploying Model Training on Multiple GPUs ...')
print("Train Frame List Shape: ", np.array(train_frame_list).shape)
print("Train Frame List Member Shape: ", np.array(train_frame_list[0]).shape)
print("Class List Train Shape: ", np.array(class_list_train).shape)
print("Test Frame List Shape: ", np.array(test_frame_list).shape)
print("Test Frame List Member Shape: ", np.array(test_frame_list[0]).shape)
print("Class List Test Shape: ", np.array(class_list_test).shape)
#print(len(test_frame_list)//(no_frame * batch_size))

# ------------- Implementing Checkpoint Restoration -------------
# if there is no specific model checkpoint supplied, then initialize the network and compile the model
if args["model"] is None:
    print("[INFO] compiling model...")
    model = c3d_model()

# otherwise, we're using a checkpoint model
else:
    # load the checkpoint from disk
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])

#model = model_from_json(open('/home/student/usama_lahore/Ramna/Results/6-epochs-29%acc-overfit/ucf_crime.json', 'r').read())

#model=c3d_model()
#model.load_weights('/home/student/usama_lahore/Ramna/Results/6-epochs-29%acc-overfit/ucf_crime_weights_file.h5')

# Replicates `model` on n GPUs:
#model_multi = multi_gpu_model(model_1, gpus=2)


model.summary()


#original optimizer code
#le=0.001
#opt = SGD(lr=le, momentum=0.009, decay=le)
##model_1.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#using the same optimizer as waqas sultani 
#opt=Adagrad(lr=0.01, epsilon=1e-08)

#current optimizer
opt=Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#train_frame_list1,class_list_train1,test_frame_list1,class_list_test1= train_test_split(generate_data(train_frame_list,class_list_train,batch_size),test_size=0.33, random_state=42)
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


#----------------------------- Training the Model -----------------------------

# construct the set of callbacks
#callbacks = [
#    EpochCheckpoint(args["checkpoints"], every=1,
#        startAt=args["start_epoch"])]

#filepath = '/home/student/usama_lahore/Ramna/trained_images/train/ucf_crime_weights_file.h5'
#checkpoint = ModelCheckpoint(filepath, monitor='val_accuraccy', verbose=1, save_best_only=True, mode = 'max')
#callbacks_list = [checkpoint]


# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
#filepath = '/home/jupyter/Drive_2/Event_Detection/Saved_Model/weights_ucf_complete_16mp.h5'
#mc = ModelCheckpoint(filepath, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
trainingResults = open('training_results.txt', 'w')
print ('\n\n\n\n*******************************************************************************************')
print ("ALERT >> Model Training Started ....")
trainingHistory = model.fit_generator(generate_data(train_frame_list,class_list_train,batch_size), 
                    steps_per_epoch=floor(len(train_frame_list)/(no_frame * batch_size)),
                    epochs=epochs,
                    validation_data=generate_data(test_frame_list,class_list_test,batch_size),
                    validation_steps=floor(len(test_frame_list)/ (no_frame * batch_size )),
                    verbose=1,callbacks=[es])


iteration = 1
for epoch in trainingHistory.history['val_loss']:
    trainingResults.write("Validation Loss for Epoch #" + str(iteration) + " : " + str(epoch) + "\n")
    iteration += 1

iteration = 1
for epoch in trainingHistory.history['val_accuracy']:
    trainingResults.write("Validation Accuracy for Epoch #" + str(iteration) + " : " + str(epoch) + "\n")
    iteration += 1
    
iteration = 1
for epoch in trainingHistory.history['loss']:
    trainingResults.write("Training Loss for Epoch #" + str(iteration) + " : " + str(epoch) + "\n")
    iteration += 1
    
iteration = 1
for epoch in trainingHistory.history['accuracy']:
    trainingResults.write("Training Accuracy for Epoch #" + str(iteration) + " : " + str(epoch) + "\n")
    iteration += 1

trainingResults.close()


# plot training history
plt.plot(trainingHistory.history['loss'], label='train_loss')
plt.plot(trainingHistory.history['val_loss'], label='validation_loss')
plt.plot(trainingHistory.history['accuracy'], label='train_acc')
plt.plot(trainingHistory.history['val_accuracy'], label='validation_acc')
plt.legend()
plt.savefig('/home/jupyter/Drive_2/Event_Detection/plot.svg')  
plt.show()


#----------------------------- Save Weights and Serialized Model to JSON -----------------------------

print ('\n\n\n\n*******************************************************************************************')
print ("ALERT >> Serializing Model to Disk ....")
model_json = model.to_json()
with open('/home/jupyter/Drive_2/Event_Detection/Saved_Model/arch_ucf_complete_16mp.json',"w") as json_file:
    json_file.write(model_json)
    json_file.close()
model.save_weights("/home/jupyter/Drive_2/Event_Detection/Saved_Model/weights_ucf_complete_16mp.h5") #Onlly Weights

# Saving/loading whole models (architecture + weights + optimizer state):
# model_1.save('/home/jupyter/Drive_2/Event_Detection/Saved_Model/ucf_hd5_arch_weights_optimizer_exp_1.h5')
model.save('/home/jupyter/Drive_2/Event_Detection/Saved_Model_1/weights_state_ucf_complete_16mp.h5', overwrite=True, include_optimizer=True)
#model.save_model
print ("ALERT >> Serialization Successful. Done ....")

#----------------------------- Model Evaluation -----------------------------

#loss,acc = model.evaluate_generator(
 #   generate_data(train_frame_list,class_list_train,batch_size),steps=floor(len(train_frame_list)/ (batch_size * no_frame)))
#print("loss_train_data",loss)
#print("accuracy_training",acc)

evaluationResults = open('evaluation_results.txt', 'w')
print ("\n\n ------ Model Evaluation Metrics ------ ", file=evaluationResults)
print (" ------ Model Evaluation Metrics ------ ", file=evaluationResults)

# Training Loss Metrics:
loss,acc= model.evaluate_generator(generate_data(train_frame_list,class_list_train,batch_size),steps=floor(len(train_frame_list)/ (batch_size * no_frame)))
print ("\n------ Training Evaluation Metrics ------", file=evaluationResults)
print("Model Training Loss: ",loss, file=evaluationResults)
print("Model Training Accuracy: ",acc, file=evaluationResults)

# Testing Loss Metrics:
loss, acc= model.evaluate_generator(generate_data(test_frame_list,class_list_test,batch_size),steps=floor(len(test_frame_list)/(batch_size * no_frame)))
print ("------ Testing Evaluation Metrics ------\n", file=evaluationResults)
print("Model Testing Loss: ",loss, file=evaluationResults)
print("Model Testing Accuracy: ",acc, file=evaluationResults)
evaluationResults.close()

