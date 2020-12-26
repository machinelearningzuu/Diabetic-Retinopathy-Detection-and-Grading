from variables import*
from util import*
from tkinter import *

from tkinter import filedialog
from inference import TestModel 
from model import DiabeticRetinopathyDetection

if not os.path.exists('data/weights/'):
    os.makedirs('data/weights/')

if not os.path.exists('data/visualization/'):
    os.makedirs('data/visualization/')

def run(TFmodel):
    if not os.path.exists(model_converter):
        keras_model = DiabeticRetinopathyDetection()
        keras_model.run()
        TFmodel.TFconverter(keras_model.model)
    TFmodel.TFinterpreter()

def preprocess_image(img_path):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = preprocessing_function(img)
    img = cv.resize(img, target_size, cv.INTER_AREA).astype(np.float32)
    return img

def process_output(TFmodel, img):
    output = TFmodel.InferenceOutput(img)
    output = output.squeeze()
    output_label = output.argmax()
    return class_dict[int(output_label)]

def get_image_path():
    root = Tk()
    root.filename =  filedialog.askopenfilename(
                                            initialdir = initial_dir, 
                                            title = "Select file",
                                            filetypes = (
                                                    ("all files","*.*"),
                                                    ("jpeg files","*.jpg"),
                                                    ("png files","*.png*")
                                                    )
                                                )
    return root.filename

def test():
    TFmodel = TestModel()
    run(TFmodel)

    img_path = get_image_path()
    img = preprocess_image(img_path)

    output = process_output(TFmodel, img)
    print(output)

if __name__ == "__main__":
    test()