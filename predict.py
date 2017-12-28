#!/usr/bin/env python
import sys
import os
import time
import subprocess as sp
import itertools
import cv2
import numpy as np
import utils

import variables
import preprocess
img_height = variables.img_height
img_width = variables.img_width
img_channels = variables.img_channels


epoch_id = 10

model = utils.get_model()

print('processing data for epoch {} ...'.format(epoch_id))
vid_path = utils.join_dir(
    variables.data_dir, 'epoch{:0>2}_front.mkv'.format(epoch_id))
assert os.path.isfile(vid_path)
cap = cv2.VideoCapture(vid_path)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

imgs_test, wheels_test = preprocess.load_data('test')
imgs_test = np.array(imgs_test)

machine_steering = []

print('performing inference...')
time_start = time.time()


machine_steering = model.predict(imgs_test, batch_size=128, verbose=0)

fps = frame_count / (time.time() - time_start)

print('completed inference, total frames: {}, average fps: {} Hz'.format(
    frame_count, round(fps, 1)))

print('performing visualization...')
utils.visualize(epoch_id, machine_steering, variables.out_dir,
                verbose=True, frame_count_limit=None)
