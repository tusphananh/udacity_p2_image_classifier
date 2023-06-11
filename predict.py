#importing necessary libraries
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import argparse
import json

# define Mandatory and Optional Arguments for the script
parser = argparse.ArgumentParser (description = "Parser of prediction script")

parser.add_argument ('image_dir', help = 'Provide path to image. Mandatory argument', type = str)
parser.add_argument ('load_dir', help = 'Provide path to checkpoint. Mandatory argument', type = str)
parser.add_argument ('--top_k', help = 'Top K most likely classes. Optional', type = int)
parser.add_argument ('--category_names', help = 'Mapping of categories to real names. JSON file name to be provided. Optional', type = str)

# a function that loads a checkpoint and rebuilds the model
def loading_model (file_path):
    # use tf.keras.models.load_model to load your saved model
    model = tf.keras.models.load_model(file_path, custom_objects={'KerasLayer':hub.KerasLayer})
    return model

def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image.numpy()

#defining prediction function
def predict(image_path, model, top_k = 5):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    processed_test_image = np.expand_dims(processed_test_image, axis=0)
    ps = model.predict(processed_test_image)
    top_k_values, top_k_indices = tf.math.top_k(ps, k=top_k)
    top_k_values = top_k_values.numpy()[0]
    top_k_indices = top_k_indices.numpy()[0]
    return top_k_values, top_k_indices

#setting values data loading
args = parser.parse_args ()
print(args)
file_path = args.image_dir

#loading JSON file if provided, else load default file name
if args.category_names:
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
else:
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)
        pass

#loading model from checkpoint provided
model = loading_model (args.load_dir)

#defining number of classes to be predicted. Default = 1
if args.top_k:
    nm_cl = args.top_k
else:
    nm_cl = 1

#calculating probabilities and classes
probs, classes = predict (file_path, model, nm_cl)
cl_names = []

#preparing class_names using mapping with class_names
for i in classes:
    print(i)
    cl_names.append(class_names[str(i+1)])

for l in range (nm_cl):
     print("Number: {}/{}.. ".format(l+1, nm_cl),
            "Class name: {}.. ".format(cl_names [l]),
            "Probability: {:.3f}..% ".format(probs [l]*100),
            )