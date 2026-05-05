pip install --upgrade keras

pip install --upgrade tensorflow

#Importing the necessary libraries

import tensorflow as tf
import albumentations as albu
import numpy as np
import gc
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import CSVLogger
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score, f1_score
import cv2
import time
import os
import h5py

from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, add, multiply, Lambda
from keras.layers import AveragePooling2D, average, UpSampling2D, Dropout
from keras.initializers import glorot_normal, random_normal, random_uniform
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda,Conv1D

from keras import backend as K
from keras.layers import BatchNormalization
from keras.models import load_model

import numpy as np
import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Reshape, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve # roc curve tools
from sklearn.model_selection import train_test_split

# Checking the number of GPUs available

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Settmgng the model parameters
import os

# Create directories
os.makedirs('ProgressFull', exist_ok=True)
os.makedirs(f'ModelSaveTensorFlow', exist_ok=True)

# Settmgng the model parameters
seed = 1
tf.random.set_seed(seed)
img_size = 256
# dataset_type = 'kvasir' # Options: kvasir/cvc-clinicdb/cvc-colondb/etis-laribpolypdb
dataset_type = 'kvasir' 
learning_rate = 1e-3
seed_value = 58800
filters = 16 # Number of filters, the paper presents the results with 17 and 34
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
initializer = tf.keras.initializers.HeNormal(seed=seed)

ct = datetime.now()

model_type = "Evo Gray U-Net"