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
Eye detection
"""
# Preliminary screening
def create_preliminary_screening_dataset(images, debug=True, visualize=True):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    if not os.path.exists("./raw eyes images"):
        os.makedirs("./raw eyes images")

    for id, image_id in images.items():
        if debug:
            print(id)
        
        for i, image_i in image_id.items():
            if debug:
                print(i)

            eyes = eye_cascade.detectMultiScale(image_i, scaleFactor = 1.001, minNeighbors = 10, minSize=[600, 600], maxSize=[2000, 2000])
            
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
                if not os.path.exists("./raw eyes images/" + str(id)):
                    os.makedirs("./raw eyes images/" + str(id))
                
                cv2.imwrite("./raw eyes images/" + str(id) + "/" + str(i) + "_" + str(count) + ".jpg", eyes_image_i)

# Labeling dataset (conjunctiva vs non-conjunctiva)
def create_labeling_dateset(images, debug=True, visualize=True):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    if not os.path.exists("./count images"):
        os.makedirs("./count images")

    count = 0
    for id, image_id in images.items():
        if debug:
            print(id)
        
        for i, image_i in image_id.items():
            if debug:
                print(i)

            eyes = eye_cascade.detectMultiScale(image_i, scaleFactor = 1.001, minNeighbors = 10, minSize=[600, 600], maxSize=[2000, 2000])
            
            # visualize the position of detected eyes
            if visualize:
                for (x,y,w,h) in eyes:
                    cv2.rectangle(image_i,(x,y),(x+w,y+h),(0, 255, 0),5)
                cv2.imshow("Eyes Detected", image_i)
                cv2.waitKey(0)
            
            # crop images of detected eyes
            for (x,y,w,h) in eyes:
                count += 1
                eyes_image_i = image_i[y:y+h, x:x+w,:]
                if visualize:
                    cv2.imshow("Eyes Detected", eyes_image_i)
                    cv2.waitKey(0)
                if not os.path.exists("./count images/" + str(id)):
                    os.makedirs("./count images/" + str(id))
                
                cv2.imwrite("./count images/" + str(id) + "/" + str(id) + "_" + str(count) + ".jpg", eyes_image_i)

"""
Conjunctiva- and non-conjunctiva images classification
"""
def train_model():
    conjunctiva_images_path = os.path.join("conjunctiva images")

    seed_list = [123, 124, 125, 126, 127]
    seed_num = 0

    train_ds = image_dataset_from_directory(
    conjunctiva_images_path,
    validation_split=0.2,
    subset="training",
    seed=seed_list[seed_num],
    image_size=(224, 224),
    batch_size=32)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    conjunctiva_images_path,
    validation_split=0.2,
    subset="validation",
    seed=seed_list[seed_num],
    image_size=(224, 224),
    batch_size=32)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

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

    for layer in base_model.layers:
        layer.trainable = False

    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    p_tb = TensorBoard(log_dir='./logs/pretraining')
    f_tb = TensorBoard(log_dir='./logs/finetuning')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[es, p_tb])

    for layer in base_model.layers[:249]:
        layer.trainable = False
    for layer in base_model.layers[249:]:
        layer.trainable = True

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[es, f_tb])

    for images, labels in val_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(str(labels[i].numpy()))
            plt.axis("off")

    for images, labels in val_ds.take(1):
        print(labels)
        y_hat = np.squeeze(model.predict(images) >= 0.5)
        print(y_hat)
        print("wrong: ", np.sum(labels != y_hat), "/", len(labels))

    model.save_weights("./models/" + str(seed_list[seed_num]) + "_" + str(0.9413) + "/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="conjunctiva prediction")
    parser.add_argument("-p", "--preliminary", help="create preliminary screening dataset", action="store_true")
    parser.add_argument("-l", "--label", help="create labeling dateset", action="store_true")
    args = parser.parse_args()

    images = import_data()
    if args.preliminary:
        create_preliminary_screening_dataset(images, debug=True, visualize=False)
    if args.label:
        create_labeling_dateset(images, debug=True, visualize=False)

    train_model()

    