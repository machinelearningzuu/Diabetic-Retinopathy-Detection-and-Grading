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
        functional_model = tf.keras.applications.VGG16(
                                                    weights = 'imagenet'
                                                        )
        functional_model.trainable = False
        inputs = functional_model.input

        x = functional_model.layers[-2].output
        x = Dense(dense_1, activation='relu')(x)
        x = Dense(dense_2, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(dense_3, activation='relu')(x)
        x = Dense(dense_3, activation='relu')(x)
        x = Dropout(keep_prob)(x)
        x = Dense(dense_4, activation='relu')(x)
        x = Dense(dense_4, activation='relu')(x)
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
        self.history = self.model.fit_generator(
                                        self.train_generator,
                                        steps_per_epoch= self.train_step,
                                        validation_data= self.test_generator,
                                        validation_steps = self.test_step,
                                        epochs=epochs,
                                        verbose=verbose
                                            )
        self.save_model()
        self.plot_metrics()

    def plot_metrics(self):
        loss_train = self.history.history['loss']
        loss_val = self.history.history['val_loss']

        loss_train = np.cumsum(loss_train) / np.arange(1,num_epoches+1)
        loss_val = np.cumsum(loss_val) / np.arange(1,num_epoches+1)

        plt.plot(np.arange(1,num_epoches+1), loss_train, 'r', label='Training loss')
        plt.plot(np.arange(1,num_epoches+1), loss_val, 'b', label='validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(loss_img)
        plt.legend()
        plt.show()

        acc_train = self.history.history['accuracy']
        acc_val = self.history.history['val_accuracy']

        acc_train = np.cumsum(acc_train) / np.arange(1,num_epoches+1)
        acc_val = np.cumsum(acc_val) / np.arange(1,num_epoches+1)

        plt.plot(np.arange(1,num_epoches+1), acc_train, 'r', label='Training Accuracy')
        plt.plot(np.arange(1,num_epoches+1), acc_val, 'b', label='validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.savefig(acc_img)
        plt.legend()
        plt.show()

    def save_model(self):
        self.model.save(model_weights)
        print("Diabetic Retinopathy Detection Model Saved")

    def load_model(self):
        K.clear_session()
        self.model = load_model(model_weights)
        print("Diabetic Retinopathy Detection Model Loaded")

    def run(self):
        if os.path.exists(model_weights):
            self.load_model()
        else:
            self.model_conversion()
            self.train()