import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import cv2
import glob
import variables, preprocess
import pickle
import nn_model as nn
from keras.models import load_model

np.random.seed(0)


data_dir = variables.data_dir
model_dir = variables.model_dir

img_height = variables.img_height
img_width = variables.img_width
img_channels = variables.img_channels
batch_size = variables.batch_size


if not variables.data_loaded:
    imgs_data, wheels_data = preprocess.load_data('train', 'YUV', flip=False)
    imgs_data_normal, wheels_data_normal = preprocess.load_data('train', flip=False)
    imgs_test, wheels_test = preprocess.load_data('test', 'YUV')
    print("Data loaded.")

    # Save train data into a pickle file.
    pickle.dump(imgs_data, open('imgs_train.p', 'wb'))
    pickle.dump(wheels_data, open('wheels_train.p', 'wb'))

    pickle.dump(imgs_data_normal, open('imgs_normal_train.p', 'wb'))
    pickle.dump(wheels_data_normal, open('wheels_normal_train.p', 'wb'))

    # Save test data into a pickle file.
    pickle.dump(imgs_test, open('imgs_test.p', 'wb'))
    pickle.dump(wheels_test, open('wheels_test.p', 'wb'))
    print("Data dumped.")

else:
    imgs_data = pickle.load(open('imgs_train.p', 'rb'))
    wheels_data = pickle.load(open('wheels_train.p', 'rb'))
    imgs_data_normal = pickle.load(open('imgs_normal_train.p', 'rb'))
    wheels_data_normal = pickle.load(open('wheels_normal_train.p', 'rb'))

    imgs_test = pickle.load(open('imgs_test.p', 'rb'))
    wheels_test = pickle.load(open('wheels_test.p', 'rb'))

    print("Data loaded.")

X_train, X_val, y_train, y_val = preprocess.split_data_set(imgs_data, wheels_data, test_size=0.2, shuffle=True)

X_train_normal, X_val_normal, y_train_normal, y_val_normal = preprocess.split_data_set(imgs_data_normal, wheels_data_normal,
                                                                            test_size=0.2, shuffle=True)

print(X_train.shape[0], 'train samples, ', X_val.shape[0], 'Valid samples')
print('X_train shape:', X_train.shape, '\nX_val shape:',X_val.shape)
print('y_train shape:',y_train.shape, '\ny_val shape:',y_val.shape)


# Train the model with ordered data
base_model = nn.base_model(img_height, img_width, img_channels)
time_start = time.time()
# Fit the model
base_history = base_model.fit(imgs_data, wheels_data,
                              epochs=1,
                              batch_size=512,
                              validation_split=0.2)

total_time = time.time() - time_start
print('Training time: {}'.format(total_time))

# Test the performance on test data
test_loss= base_model.evaluate(imgs_test, wheels_test, batch_size=512)
print('Test loss is:{}'.format(test_loss))


# list all data in history
print(base_history.history.keys())
# summarize history for loss


predicted_steering = base_model.predict(imgs_test, batch_size=128, verbose=0)


def get_ground_truth(epoch_id):
    epoch_dir = variables.data_dir
    assert os.path.isdir(epoch_dir)
    steering_path = os.path.join(epoch_dir, 'epoch{:0>2}_steering.csv'.format(epoch_id))
    assert os.path.isfile(steering_path)

    rows = pd.read_csv(steering_path)
    ground_truth = list(rows.wheel.values)
    return ground_truth


ground_truth = get_ground_truth(10)

# Data type transfer
ground_truth = np.array(ground_truth)
ground_truth = np.reshape(ground_truth,(len(ground_truth),1))


prediction_error = np.subtract(predicted_steering, ground_truth)
print("Error: ", prediction_error)


plt.figure
plt.plot(predicted_steering, color='b')
plt.plot(ground_truth, color='g')
plt.ylabel('Steering angle (deg)', fontsize=11)
plt.xlabel('Frame no.', fontsize=11)
plt.legend(['Predicted steering', 'Ground truth value'], loc='upper left')
plt.xlim((0,3500))
#plt.xticks(np.arange(0, 21, 5))
plt.grid()
plt.savefig(img_dir + "/result2.png", dpi=300)


plt.figure
plt.plot(prediction_error, color='r')
plt.ylabel('Prediction Error', fontsize=11)
plt.xlabel('Frame no.', fontsize=11)
plt.xlim((0,2700))
plt.grid()
plt.savefig(img_dir + "/result_error.png", dpi=300)

plt.figure
plt.plot(prediction_error, color='r')
plt.ylabel('Prediction Error', fontsize=11)
plt.xlabel('Frame no.', fontsize=11)
plt.xlim((0,3500))
plt.grid()
plt.savefig(img_dir + "/result_error2.png", dpi=300)
