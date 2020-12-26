from variables import*
from util import*

from inference import TestModel 
from model import DiabeticRetinopathyDetection

if not os.path.exists('data/weights/'):
    os.makedirs('data/weights/')

if not os.path.exists('data/visualization/'):
    os.makedirs('data/visualization/')

def run(TFmodel):
    keras_model = DiabeticRetinopathyDetection()
    keras_model.run()

    if not os.path.exists(model_converter):
        TFmodel.TFconverter(keras_model.model)
    TFmodel.TFinterpreter()

def preprocess_image(img_path):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = preprocessing_function(img)
    img = cv.resize(img, target_size, cv.INTER_AREA).astype(np.float32)
    return img

if __name__ == "__main__":
    TFmodel = TestModel()
    run(TFmodel)

    img_path = 'data/Train Data/2/0c7e82daf5a0.png'
    
    img = preprocess_image(img_path)
    output = TFmodel.InferenceOutput(img)