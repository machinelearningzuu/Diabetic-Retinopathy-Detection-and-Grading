import os
batch_size = 16
test_size = 12
color_mode = 'rgb'
width = 224
height = 224
target_size = (width, height)
input_shape = (width, height, 3)
shear_range = 0.2
zoom_range = 0.2
rotation_range = 20
shift_range = 0.2
dense_1 = 1000
dense_2 = 512
dense_3 = 256
dense_4 = 64
num_classes = 5
epochs = 5
verbose = 1
test_split = 0.1
seed = 42
learning_rate = 0.0001

# data directories and model paths
image_dir = 'data/images/'
label_dir = 'data/image labels/'
label_csv = 'data/Labels.csv'
model_weights = "data/weights/retinopathy_model.h5"
model_converter = "data/weights/model.tflite"

# {0: 6150, 1: 588, 2: 1283, 4: 166, 3: 221}

extension = '.jpeg'
images_per_class = 1000