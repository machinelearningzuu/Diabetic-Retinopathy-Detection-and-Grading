import os
import pickle
import logging
import pathlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

logging.getLogger('tensorflow').disabled = True

from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json, Sequential, Model, load_model
from tensorflow.keras.layers import Activation, Dense, Input, Flatten, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

from util import *
from variables import *

np.random.seed(seed)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("\nNum GPUs Available: {}\n".format(len(physical_devices)))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class DiabeticRetinopathyDetection(object):
    def __init__(self):
        train_generator, test_generator = image_data_generator()
        self.test_generator = test_generator
        self.train_generator = train_generator
        self.test_step = self.test_generator.samples // test_size
        self.train_step = self.train_generator.samples // batch_size

    def model_conversion(self):
        functional_model = tf.keras.applications.MobileNetV2(
                                                    weights = None, 
                                                    include_top=False, 
                                                    input_shape=input_shape
                                                             )
        functional_model.trainable = False
        inputs = functional_model.input

        x = functional_model.layers[-2].output
        # x = Dense(dense_1, activation='relu')(x)
        # x = Dense(dense_2, activation='relu')(x)
        # x = BatchNormalization()(x)
        # x = Dense(dense_3, activation='relu')(x)
        # x = Dense(dense_3, activation='relu')(x)
        # x = Dense(dense_4, activation='relu')(x)
        # x = Dense(dense_4, activation='relu')(x)
        # outputs = Dense(num_classes, activation='softmax')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)

        model = Model(
                inputs =inputs,
                outputs=outputs
                    )
        self.model = model
        self.model.summary()

    def train(self):
        self.model.compile(
                          optimizer='rmsprop',
                          loss='categorical_crossentropy',
                          metrics=['accuracy']
                          )
        self.model.fit_generator(
                          self.train_generator,
                          steps_per_epoch= self.train_step,
                          validation_data= self.test_generator,
                          validation_steps = self.test_step,
                          epochs=epochs,
                          verbose=verbose
                        )

    def save_model(self):
        self.model.save()
        print("Diabetic Retinopathy Detection Model Saved")

    def load_model(self):
        K.clear_session() #clearing the keras session before load model
        self.feature_model = load_model(model_weights)
        print("Diabetic Retinopathy Detection Model Loaded")

    def run(self):
        if os.path.exists(model_weights):
            print("Loading the model !!!")
            self.load_model()
        else:
            print("Training the model !!!")
            self.model_conversion()
            self.train()

model = DiabeticRetinopathyDetection()
model.run()