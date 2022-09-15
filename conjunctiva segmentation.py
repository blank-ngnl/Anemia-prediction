import pandas as pd
import numpy as np
import os
import argparse
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

"""
Import data
"""
def import_data():
    metadata_path = os.path.join("D:", "OneDrive_1_5-26-2022", "pid2625_charlesjohnson.xlsx")
    data_path = os.path.join("D:", "OneDrive_1_5-26-2022", "pid2625", "pid2625")
    metadata = pd.read_excel(metadata_path)

    images = {}

    for id in metadata["record"].unique():
        for eyeimage_i in metadata.loc[metadata['record'] == id]["field_name"]:
            if eyeimage_i in ["eyeimage1", "eyeimage2"]:
                file_name = metadata[(metadata["record"] == id) & (metadata["field_name"] == eyeimage_i)]["stored_name"].values[0]
                #print(id, eyeimage_i, file_name)
                image_path = os.path.join(data_path, file_name)
                img = cv2.imread(image_path)
                #cv2.imshow("preview", img)
                #cv2.waitKey(1000)
                #cv2.destroyAllWindows()
                
                if not images.get(id):
                    images[id] = {}
                images[id][eyeimage_i] = img

    return images

"""
Finding potential conjunctiva images
"""
def create_potential_conjunctiva_dataset(images, debug=True, visualize=True):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    if not os.path.exists("./potential eyes images"):
        os.makedirs("./potential eyes images")

    for id, image_id in images.items():
        if debug:
            print(id)
        
        for i, image_i in image_id.items():
            if debug:
                print(i)

            eyes = eye_cascade.detectMultiScale(image_i, scaleFactor = 1.0001, minNeighbors = 1, minSize=[600, 600], maxSize=[2000, 2000])
            
            # visualize the position of detected eyes
            if visualize:
                for (x,y,w,h) in eyes:
                    cv2.rectangle(image_i,(x,y),(x+w,y+h),(0, 255, 0),5)
                cv2.imshow("Eyes Detected", image_i)
                cv2.waitKey(0)
            
            # crop images of detected eyes
            count = 0
            for (x,y,w,h) in eyes:
                count += 1
                eyes_image_i = image_i[y:y+h, x:x+w,:]
                if visualize:
                    cv2.imshow("Eyes Detected", eyes_image_i)
                    cv2.waitKey(0)
                if not os.path.exists("./potential eyes images/" + str(id)):
                    os.makedirs("./potential eyes images/" + str(id))
                
                cv2.imwrite("./potential eyes images/" + str(id) + "/" + str(i) + "_" + str(count) + ".jpg", eyes_image_i)

"""
conjunctiva segmentation
"""
def conjunctiva_segmentation():
    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model = tf.keras.Sequential([
    # Add the preprocessing layers you created earlier.
    tf.keras.layers.Resizing(224, 224),
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    model
    ])

    model.load_weights("./models/123_0.9413/")

    if not os.path.exists("./detected eyes images"):
        os.makedirs("./detected eyes images")
        
    for folder in os.listdir("./potential eyes images"):
        for image in os.listdir(os.path.join("./potential eyes images", folder)):
            #print(image)
            img = cv2.imread(os.path.join("./potential eyes images", folder, image))
            #cv2.imshow("Eye detected", img)
            #cv2.waitKey(0)
            img = np.expand_dims(img, axis=0)
            y_hat = model.predict(img, verbose=0)
            #print(y_hat)
            
            if y_hat < 0.5:
                if not os.path.exists(os.path.join("./detected eyes images", folder)):
                    os.makedirs(os.path.join("./detected eyes images", folder))
                    
                cv2.imwrite(os.path.join("./detected eyes images", folder, image), img[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="conjunctiva prediction")
    parser.add_argument("-p", "--potential", help="create potential conjunctiva dataset", action="store_true")
    args = parser.parse_args()

    images = import_data()
    if args.potential:
        create_potential_conjunctiva_dataset(images, debug=True, visualize=False)

    conjunctiva_segmentation()