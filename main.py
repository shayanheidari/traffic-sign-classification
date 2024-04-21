import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import os
import random
import shutil
import zipfile


input_base_path = 'dataset'
testing_folder = 'traffic_Data/TEST'
classes = pd.read_csv(os.path.join(input_base_path, 'labels.csv'))
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
test_dir = os.path.join(input_base_path, testing_folder)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size = IMG_SIZE,
                                             batch_size = BATCH_SIZE)

model = tf.keras.models.load_model("./colab_vgg16.h5")

result = model.evaluate(test_data)
print(f'result: {result}')
