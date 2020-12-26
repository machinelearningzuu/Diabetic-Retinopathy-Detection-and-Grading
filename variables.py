import os
batch_size = 32
test_size = 16
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
epochs = 2
verbose = 1
test_split = 0.15
seed = 42
keep_prob = 0.5
learning_rate = 0.0001

# data directories and model paths
image_dir = 'data/images/'
acc_img = 'data/visualization/accuracy.png'
loss_img = 'data/visualization/loss.png'
label_dir = 'data/Train Data/'
label_csv = 'data/Labels.csv'
model_weights = "data/weights/retinopathy_model.h5"
model_converter = "data/weights/model.tflite"

initial_dir = '/home/isuru1997/Projects and Codes/DBS projects/Diabetic Retinopathy Detection and Grading/data/Train Data/'
extension = '.jpeg'
images_per_class = 1000

class_dict = {
    0 : 'No Diabetic Retinopathy',
    1 : 'Mild',
    2 : 'Moderate',
    3 : 'Severe',
    4 : 'Proliferative Diabetic Retinopathy',
    }