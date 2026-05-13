import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/deepfake_model.h5")

def predict(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128,128))
    img = img/255.0
    img = img.reshape(1,128,128,3)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        print("FAKE")
    else:
        print("REAL")

predict("test.jpg")