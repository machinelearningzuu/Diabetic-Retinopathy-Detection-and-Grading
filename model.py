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
        # train_generator, validation_generator, test_generator = image_data_generator()
        # self.test_generator = test_generator
        # self.train_generator = train_generator
        # self.validation_generator = validation_generator
        # self.train_step = self.train_generator.samples // batch_size
        # self.validation_step = self.validation_generator.samples // valid_size
        # self.test_step = self.test_generator.samples // batch_size

        Ytrain, Xtrain, Ytest , Xtest = load_images()
        self.Ytrain = Ytrain
        self.Xtrain = Xtrain
        self.Ytest = Ytest
        self.Xtest = Xtest

        print("Xtrain shape : {}".format(Xtrain.shape))
        print("Ytrain shape : {}".format(Ytrain.shape))
        print("Xtest  shape : {}".format(Xtest.shape))
        print("Ytest  shape : {}".format(Ytest.shape))

    def model_conversion(self):
        functional_model = tf.keras.applications.MobileNetV2(
                                                    weights="imagenet"
                                                             )
        functional_model.trainable = False
        inputs = functional_model.input

        x = functional_model.layers[-2].output
        x = Dense(dense_1, activation='relu')(x)
        x = Dense(dense_1, activation='relu')(x)
        x = Dense(dense_2, activation='relu')(x)
        x = Dense(dense_2, activation='relu')(x)
        outputs = Dense(len(get_class_names()), activation='softmax')(x)

        model = Model(
                inputs =inputs,
                outputs=outputs
                    )
        self.model = model
        self.model.summary()

    def train(self):
        self.model.compile(
                          optimizer=Adam(learning_rate),
                          loss='categorical_crossentropy',
                          metrics=['accuracy']
                          )
        self.model.fit_generator(
                          self.train_generator,
                          steps_per_epoch= self.train_step,
                          validation_data= self.validation_generator,
                          validation_steps = self.validation_step,
                          epochs=epochs,
                          verbose=verbose
                        )

    def deep_ml_model(self):
        functional_model = tf.keras.applications.MobileNetV2(
                                                    weights="imagenet"
                                                             )
        pred = functional_model.predict(self.Xtrain)
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(pred, self.Ytrain)

        score = model.score(pred, self.Ytrain)
        print(score)

    def save_model(self):
        self.model.save(model_weights)
        print("Diabetic Retinopathy Detection Model Saved")

    def loading_model(self):
        K.clear_session() #clearing the keras session before load model
        self.feature_model = load_model(model_weights)
        print("Diabetic Retinopathy Detection Model Loaded")

    def Evaluation(self):
        Predictions = self.model.predict_generator(self.test_generator,steps=self.test_step)
        P = np.argmax(Predictions,axis=1)
        loss , accuracy = self.model.evaluate_generator(self.test_generator, steps=self.test_step)
        print("test loss : ",loss)
        print("test accuracy : ",accuracy)

model = DiabeticRetinopathyDetection()
# model.model_conversion()
# model.train()
model.deep_ml_model()