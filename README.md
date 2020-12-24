# Diabetic-Retinopathy-Detection-and-Grading

The objective of the task is to create an automated analysis system capable of assigning a score based on the diabetic retinopathy scale provided.

Diabetic retinopathy is the leading cause of blindness in the working-age population of the developed world. It is estimated to affect over 93 million people.Currently, detecting DR is a time-consuming and manual process that requires a trained clinician to examine and evaluate digital color fundus photographs of the retina. By the time human readers submit their reviews, often a day or two later, the delayed results lead to lost follow up, miscommunication, and delayed treatment.

Clinicians can identify DR by the presence of lesions associated with the vascular abnormalities caused by the disease. While this approach is effective, its resource demands are high. The expertise and equipment required are often lacking in areas where the rate of diabetes in local populations is high and DR detection is most needed. As the number of individuals with diabetes continues to grow, the infrastructure needed to prevent blindness due to DR will become even more insufficient.

The need for a comprehensive and automated method of DR screening has long been recognized, and previous efforts have made good progress using image classification, pattern recognition, and machine learning. With color fundus photography as input, the goal of this competition is to push an automated detection system to the limit of what is possible – ideally resulting in models with realistic clinical potential. The winning models will be open sourced to maximize the impact such a model can have on improving DR detection.

# Dataset

This dataset provided with a large set of high-resolution retina images taken under a variety of imaging conditions. A left and right field is provided for every subject. Images are labeled with a subject id as well as either left or right

A clinician has rated the presence of diabetic retinopathy in each image on a scale of 0 to 4, according to the following scale:

    0 - No DR
    1 - Mild
    2 - Moderate
    3 - Severe
    4 - Proliferative DR
    
# Methodology
  - Data Preprocessing 
  - Create Data generator for augmentation
  - Using Deep Learning and Machine Learning classfication identify diabetic retinopathy scale
  - Inference the model using tensorflow lite 
  
# Techniques

  - Supervised Deep Learning Classification
  - Supervised Machine Learning Classification
  - Artificial Neural Networks
  - K-Nearest Neighbour
  - Tensorflow Lite Inference
# Tools

* TensorFlow - Deep Learning Model
* pandas - Data Extraction and Preprocessing
* numpy - numerical computations
* scikit learn - Advanced preprocessing and Machine Learning Models

### Installation

Install the dependencies and conda environment

```sh
$ conda create -n envname python=python_version
$ activate envname 
$ conda install -c anaconda tensorflow-gpu
$ conda install -c anaconda pandas
$ conda install -c anaconda matplotlib
$ conda install -c anaconda scikit-learn
```
