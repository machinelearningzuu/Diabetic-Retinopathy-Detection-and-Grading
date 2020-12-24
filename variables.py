import os
batch_size = 8
valid_size = 6
color_mode = 'rgb'
width = 224
height = 224
target_size = (width, height)
input_shape = (width, height, 3)
shear_range = 0.2
zoom_range = 0.3
rotation_range = 30
shift_range = 0.2
mean = 0
std = 255.0
dense_1 = 1000
dense_2 = 512
dense_3 = 256
dense_4 = 64
num_classes = 5
epochs = 15
verbose = 1
val_split = 0.15
seed = 1234
learning_rate = 0.0001

# data directories and model paths
test_data_path = 'data/weights/Test_data.npz'
train_data_path = 'data/weights/Train_data.npz'
train_dir = 'data/Train images/'
test_dir = 'data/Test images/'
train_label_csv = 'data/Training Labels.csv'
test_label_csv = 'data/Testing Labels.csv'
model_weights = "data/weights/retinopathy_model.h5"
model_converter = "data/weights/model.tflite"