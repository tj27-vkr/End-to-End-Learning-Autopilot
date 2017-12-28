
import os
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

batch_size = 128
img_height = 66
img_width = 200
img_channels = 3

flags.DEFINE_integer('img_h', 66, 'The image height.')
flags.DEFINE_integer('img_w', 200, 'The image width.')
flags.DEFINE_integer('img_c', 3, 'The number of channels.')

np.random.seed(42)

data_dir = os.path.abspath('./data')
out_dir = os.path.abspath('./output')
model_dir = os.path.abspath('./models')

read_model = False

data_loaded = False
