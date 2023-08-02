import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from tkinter.filedialog import askopenfilename
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
import keras.utils as image
import tensorflow
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


images = "Garbage classification"

# set desired image size
resize = (32, 32)
categories = {"cardboard": 0, "glass": 1, "metal": 2, "paper": 3, "plastic": 4, "trash": 5}
shape = (32, 32, 3)


def data_prep():
    train = ImageDataGenerator(rescale=1. / 255,
                               shear_range=0.1,
                               zoom_range=0.1,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               horizontal_flip=True,
                               vertical_flip=True,
                               validation_split=0.1)
    test = ImageDataGenerator(rescale=1. / 255, validation_split=0.1)

    train_gen = train.flow_from_directory(images,
                                          target_size=resize,
                                          batch_size=200,
                                          class_mode='categorical',
                                          subset='training',
                                          seed=0)
    test_gen = test.flow_from_directory(images,
                                        target_size=resize,
                                        batch_size=200,
                                        class_mode='categorical',
                                        subset='validation',
                                        seed=0)
    return train_gen, test_gen


train_gen, test_gen = data_prep()


def create_model(train_gen, test_gen):
    model = Sequential([
        Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=shape),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(6, activation='softmax')])
    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=['accuracy'])
    history = model.fit_generator(generator=train_gen, epochs=100, validation_data=test_gen)
    return model, history


model, history = create_model(train_gen, test_gen)

scores = model.evaluate(train_gen, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

pd.DataFrame(history.history).plot()


def evaluate_model(model):
    x_test, y_test = test_gen.next()
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    target_names = list(categories.keys())
    print(classification_report(y_test, y_pred, target_names=target_names))

    return y_test, y_pred


y_test, y_pred = evaluate_model(model)

cm = confusion_matrix(y_test, y_pred)


def confusion_matrix(cm, classes, normalize=False, title="Matrix"):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    f = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], f), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel("True Labels", fontweight="bold")
    plt.xlabel("Predicted Labels", fontweight="bold")


confusion_matrix(cm, categories.keys())
plt.show()

while True:
    retry = input("Would you like to analyze a picture? y/n?")
    if retry == "n":
        quit()
    if retry == "y":
        def test_model(path):
            img = image.load_img(path, target_size=resize)
            img = image.img_to_array(img, dtype=np.uint8)
            img = np.array(img) / 255.0
            p = model.predict(img.reshape(1, 32, 32, 3))
            prediction = np.argmax(p[0])
            return img, p, prediction


        select_photo = askopenfilename()  # show a dialog box and return the path to the selected file
        img = cv2.imread(select_photo)
        img = cv2.resize(img, resize)
        new_labels = {0: "cardboard", 1: "glass", 2: "metal", 3: "paper", 4: "plastic", 5: "trash"}
        # LOAD IMAGE TO TEST HOW ACCURATE THE MODEL IS
        img, p, prediction = test_model(select_photo)


        def predict(img, p, prediction):
            plt.axis("off")
            plt.imshow(img.squeeze())
            plt.title("Predicted Class: " + str(
                new_labels[prediction]))
            plt.imshow(img)
            plt.show()
    predict(img, p, prediction)
