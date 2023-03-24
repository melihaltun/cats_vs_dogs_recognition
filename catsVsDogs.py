# @fileName catsVsDogs.py
# @author Melih Altun @2023

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import random
import glob
import warnings
import os
import shutil

warnings.simplefilter(action='ignore', category=FutureWarning)
#%matplotlib inline

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('Num GPUs Available: ', len(physical_devices))
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.chdir('./cats_vs_dogs')
if os.path.isdir('./train/dog') is False:
    os.makedirs('./train/dog')
    os.makedirs('./train/cat')
    os.makedirs('./test/dog')
    os.makedirs('./test/cat')
    os.makedirs('./valid/dog')
    os.makedirs('./valid/cat')

    for c in random.sample(glob.glob('cat*'), 600):
        shutil.move(c, './train/cat')
    for c in random.sample(glob.glob('dog*'), 600):
        shutil.move(c, './train/dog')
    for c in random.sample(glob.glob('cat*'), 120):
        shutil.move(c, './valid/cat')
    for c in random.sample(glob.glob('dog*'), 120):
        shutil.move(c, './valid/dog')
    for c in random.sample(glob.glob('cat*'), 60):
        shutil.move(c, './test/cat')
    for c in random.sample(glob.glob('dog*'), 60):
        shutil.move(c, './test/dog')

os.chdir('../')

train_path = './cats_vs_dogs/train'
valid_path = './cats_vs_dogs/valid'
test_path = './cats_vs_dogs/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10, shuffle=False)

assert train_batches.n == 1200
assert valid_batches.n == 240
assert test_batches.n == 120

assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 2

imgs, labels = next(train_batches)

def plotImages(images_arr):
    fig, axes = plt.subplots(1,10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#plotImages(imgs)
#print(labels)

vgg16_model = tf.keras.applications.vgg16.VGG16()

vgg16_model.summary()


type(vgg16_model)

model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

model.summary()

for layer in model.layers:
    layer.trainable = False

model.add(Dense(units=2, activation='softmax'))

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches, validation_data=valid_batches, epochs=5) #, verbose=2)

predictions = model.predict(x=test_batches, verbose=0)

test_batches.num_classes

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

test_batches.class_indices

cm_plot_labels = ['cat', 'dog']

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalizes Confusion Matrix")
    else:
        print("Confusion Matrix, without normalization")
    print(cm)
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
