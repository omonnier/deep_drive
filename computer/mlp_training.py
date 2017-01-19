import cv2
import numpy as np
import glob
from commonDeepDriveDefine import *


print 'Loading training data...'
e0 = cv2.getTickCount()

# load training data
image_array  = np.zeros((1, 38400), dtype=np.float32)
angle_array  = np.zeros(1, dtype=np.uint8)
train_labels = np.zeros((0, NN_OUTPUT_NUMBER), dtype=np.float32)
eye_array    = np.eye  (NN_OUTPUT_NUMBER, dtype=np.float32)

print 'number of output layer = ' + str(NN_OUTPUT_NUMBER)


training_data = glob.glob('training_data/*.npz')

for single_npz in training_data:
    with np.load(single_npz) as data:
        print data.files
        train_temp = data['train']
        angle_array_temp = data['train_labels']
        print train_temp.shape
        print angle_array_temp.shape
    image_array = np.vstack((image_array, train_temp))
    angle_array = np.vstack((angle_array, angle_array_temp))


# Build training labels from 0 .. N  based on angle_array
for angle in angle_array:
    angle = int (round((((angle + MAX_ANGLE) / STEP_REPLAY )),0))
    train_labels = np.vstack ((train_labels, eye_array[angle]))

# Remove first element. 
train = image_array[1:, :]
train_labels = train_labels[1:, :]

print train.shape
print train_labels.shape

e00 = cv2.getTickCount()
time0 = (e00 - e0)/ cv2.getTickFrequency()
print 'Loading image duration:', time0

# set start time
e1 = cv2.getTickCount()

# create MLP
layer_sizes = np.int32([38400, 32, NN_OUTPUT_NUMBER])
model = cv2.ANN_MLP()
model.create(layer_sizes)
criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 0.0001)
criteria2 = (cv2.TERM_CRITERIA_COUNT, 100, 0.001)
params = dict(term_crit = criteria,
               train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,
               bp_dw_scale = 0.001,
               bp_moment_scale = 0.0 )

print 'Training MLP ...'
num_iter = model.train(train, train_labels, None, params = params)

# set end time
e2 = cv2.getTickCount()
time = (e2 - e1)/cv2.getTickFrequency()
print 'Training duration:', time

# save param
model.save('mlp_xml/mlp.xml')

print 'Ran for %d iterations' % num_iter

ret, resp = model.predict(train)
prediction = resp.argmax(-1)
print 'Prediction:', prediction
true_labels = train_labels.argmax(-1)
print 'True labels:', true_labels

print 'Testing...'
train_rate = np.mean(prediction == true_labels)
print 'Train rate: %f:' % (train_rate*100)

