import os
import shutil
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from collections import Counter
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from variables import*

def preprocessing_function(img):
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def get_class_names():
    df = pd.read_csv(label_csv)
    grade = df['level'].values 
    return list(set(grade))

def move_image(source_path, destination_path):
    Path(source_path).rename(destination_path)

def copy_image(source_path, destination_path):
    shutil.copyfile(source_path, destination_path)

def create_sub_directories():
    classes = get_class_names() 
    for i in classes:
        class_dir = os.path.join(label_dir, str(i))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

def dir_exists():
    classes = get_class_names()
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    labels = list(map(int, os.listdir(label_dir)))

    classes.sort()
    labels.sort()

    if labels == classes:
        return True 
    else:
        return False

def get_class_data(label, grades, image_names):
    class_grades = grades[grades == label]
    class_images = image_names[grades == label]

    if len(class_grades) > images_per_class:
        random_idxs = np.random.choice(len(class_grades), images_per_class, replace=False)
        class_grades = class_grades[random_idxs]
        class_images = class_images[random_idxs]

    class_grades = class_grades.tolist()
    class_images = class_images.tolist()
    return class_grades, class_images

def class_balancing(grades, image_names):
    class_dict = dict(Counter(grades))
    balanced_labels = []
    balanced_images = []
    for label in get_class_names():
        class_grades, class_images = get_class_data(label, grades, image_names)
        balanced_labels += class_grades
        balanced_images += class_images
    balanced_labels = np.array(balanced_labels)
    balanced_images = np.array(balanced_images)
    return balanced_labels, balanced_images

def extract_images():
    df = pd.read_csv(label_csv)
    grades = df['level'].values 
    image_names = df['image'].values 

    image_paths = os.listdir(image_dir)
    image_paths = np.array([img.split('.')[0] for img in image_paths if len(img.split('.')) > 1])

    existed_images = np.intersect1d(image_paths, image_names)
    common_idxs = np.where(np.in1d(image_names, existed_images))[0]

    image_names = image_names[common_idxs]
    grades = grades[common_idxs]

    # grades, image_names = class_balancing(grades, image_names)
    return grades, image_names

def configure_data_directory():

    create_sub_directories()
    grades, image_names = extract_images()
    image_paths = [image_name + extension for image_name in image_names]

    for image, label in zip(image_paths, grades):
        image_path = os.path.join(image_dir, image)
        destination_path = os.path.join(label_dir, str(label), image)
        copy_image(image_path, destination_path)

def preprocess_data_directories():
    if dir_exists():
        print("Train directories are already configured")
    else:
        print("Train directories are configuring")
        configure_data_directory()

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
                                    validation_split= test_split,
                                    preprocessing_function=preprocessing_function
                                    )

    train_generator = train_datagen.flow_from_directory(
                                    label_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = batch_size,
                                    classes = classes,
                                    subset = 'training',
                                    shuffle = True)

    test_generator = train_datagen.flow_from_directory(
                                    label_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = test_size,
                                    classes = classes,
                                    subset = 'validation',
                                    shuffle = True)

    return train_generator, test_generator
