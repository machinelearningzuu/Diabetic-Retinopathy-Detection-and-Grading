import os
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from collections import Counter
from sklearn.utils import shuffle, class_weight

from variables import*

def preprocessing_function(img):
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def get_class_names():
    df = pd.read_csv(train_label_csv)
    grade = df['Retinopathy grade'].values 
    return list(set(grade))

def move_image(source_path, destination_path):
    Path(source_path).rename(destination_path)

def create_sub_directories(image_dir):
    classes = get_class_names() 
    for class_ in classes:
        class_dir = os.path.join(image_dir, str(class_))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

def dir_exists(image_dir):
    classes = get_class_names()
    labels = list(map(int, os.listdir(train_dir)))

    classes.sort()
    labels.sort()

    if labels == classes:
        return True 
    else:
        return False

def configure_data_directory(image_dir, csv_dir):

    create_sub_directories(image_dir)

    df = pd.read_csv(csv_dir)
    image_names = df['Image name'].values.tolist()
    grades = df['Retinopathy grade'].values 

    image_paths = os.listdir(image_dir)
    image_paths = [img for img in image_paths if len(img.split('.')) > 1]
    image_paths.sort()

    image_paths_without_extension = [img.split('.')[0] for img in image_paths if len(img.split('.')) > 1]
    assert image_names == image_paths_without_extension, "Image count and label count not matched"

    for image, label in zip(image_paths, grades):
        image_path = os.path.join(image_dir, image)
        destination_path = os.path.join(image_dir, str(label), image)
        move_image(image_path, destination_path)

def preprocess_data_directories():
    if dir_exists(train_dir):
        print("Train directories are already configured")
    else:
        print("Train directories are configuring")
        configure_data_directory(train_dir, train_label_csv)

    if dir_exists(test_dir):
        print("Test directories are already configured")
    else:
        print("Test directories are configuring")
        configure_data_directory(test_dir, test_label_csv)

def image_data_generator():
    classes =  list(map(str, get_class_names()))
    preprocess_data_directories()
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                    rotation_range = rotation_range,
                                    shear_range = shear_range,
                                    zoom_range = zoom_range,
                                    width_shift_range=shift_range,
                                    height_shift_range=shift_range,
                                    horizontal_flip = True,
                                    validation_split= val_split,
                                    preprocessing_function=preprocessing_function
                                    )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                    preprocessing_function=preprocessing_function
                                    )


    train_generator = train_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = batch_size,
                                    classes = classes,
                                    subset = 'training',
                                    shuffle = True)

    validation_generator = train_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = valid_size,
                                    classes = classes,
                                    subset = 'validation',
                                    shuffle = True)

    test_generator = test_datagen.flow_from_directory(
                                    test_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = batch_size,
                                    classes = classes,
                                    shuffle = False)

    return train_generator, validation_generator, test_generator

def load_numpy_data(data_path, save_path, Train=True):
    if not os.path.exists(save_path):
        if Train:
            print("Train Images Saving")
        else:
            print("Test Images Saving")
        images = []
        classes = []
        url_strings = []
        image_folders = os.listdir(data_path)
        for label in list(image_folders):
            label_dir = os.path.join(data_path, label)
            label_images = []
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                img = cv.imread(img_path)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = preprocessing_function(img)
                img = cv.resize(img, target_size, cv.INTER_AREA).astype(np.float32)

                images.append(img)
                classes.append(int(label))

        images = np.array(images).astype('float32')
        classes = np.array(classes).astype('float32')
        np.savez(save_path, name1=images, name2=classes)
    else:
        data = np.load(save_path, allow_pickle=True)
        images = data['name1']
        classes = data['name2']

        if Train:
            print("Train Images Loading")
        else:
            print("Test Images Loading")

    classes, images = shuffle(classes, images)
    return classes, images

def load_images():
    Ytrain, Xtrain = load_numpy_data(train_dir, train_data_path)
    Ytest , Xtest  = load_numpy_data(test_dir, test_data_path, False)
    return Ytrain, Xtrain, Ytest , Xtest