import tensorflow as tf
from tensorflow.keras import layers, models
import os, cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 128

def load_data(path):
    data, labels = [], []
    for label in ["real", "fake"]:
        folder = os.path.join(path, label)
        class_num = 0 if label == "real" else 1
        
        for img in os.listdir(folder):
            try:
                img_path = os.path.join(folder, img)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                data.append(image)
                labels.append(class_num)
            except:
                pass
    
    return np.array(data)/255.0, np.array(labels)

X, y = load_data("dataset/")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

model.save("model/deepfake_model.h5")

print("DONE TRAINING")