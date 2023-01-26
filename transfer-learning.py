#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import densenet
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# In[2]:


# Load and scale the data
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0


# In[3]:


# Split the data into test and validation data
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.7)


# In[4]:


print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# In[5]:


# Dataset class labels
labels =  ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 
           'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 
           'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard', 
           'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 
           'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 
           'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 
           'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 
           'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 
           'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

# View more images in a grid format
# Define the dimensions of the plot grid 
W_grid = 10
L_grid = 4

fig, axes = plt.subplots(L_grid, W_grid, figsize = (17,7))
axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array
X_train
n_train = len(X_train) # get the length of the train dataset

# Select a random number from 0 to n_train
for i in np.arange(0, W_grid * L_grid): 

    # Select a random number
    index = np.random.randint(0, n_train)
    # read and display an image with the selected index    
    axes[i].imshow(X_train[index,1:])
    label_index = int(y_train[index])
    label_index
    axes[i].set_title(labels[label_index], fontsize = 8)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)


# In[7]:


# print first 10 labels before encoding
print("before encoding:")
print(y_train[:10])

# Convert labels (target variable) to one hot encoding matrix
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)
y_valid = to_categorical(y_valid, 100)

# print first 10 labels after encoding
print("after encoding:")
print(y_train[:10])


# In[7]:


# Model building function
def model(input_shape, n_classes):
    
# Build model from DenseNet. 
# 5 Cnv layers, 2 Dense layers.
    base = densenet.DenseNet121(input_shape=input_shape,
                                      weights="imagenet",
                                      include_top=False,
                                      pooling='avg')

    for layer in base.layers[:-5]:
        layer.trainable = False

    for layer in base.layers[-5:]:
        layer.trainable = True

    x = base.output

    x = Dense(64)(x)
    x = Activation('relu')(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Build the model
model = model(input_shape=(32, 32, 3), n_classes=len(labels))
# get summary
model.summary()


# In[8]:


# Generate augmented images
datagen = ImageDataGenerator(
    height_shift_range=0.2,
    width_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=40,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
)

datagen.fit(X_train)


# In[9]:


# Set checkpoint
model_checkpointer = ModelCheckpoint('cifar100_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
# Train model
history = model.fit(datagen.flow(X_train, y_train, batch_size=64, shuffle=True), validation_data=(X_valid, y_valid), epochs=30, verbose=1,  
               callbacks=[EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10), model_checkpointer])


# In[10]:


# Show Loss and Accuracy Plots
fig, ax = plt.subplots(2, 1)

ax[0].plot(history.history['loss'], color='b', label="Training Loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation Loss",axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training Accuracy")
ax[1].plot(history.history['val_accuracy'], color='r', label="Validation Accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# In[11]:


# Evaluate model
_, evaluation_score = model.evaluate(X_test, y_test)
print(f'Evaluation Score: {int(evaluation_score * 100)} %')


# In[12]:


#Classification report
pred = np.argmax(model.predict(X_test), axis=1)
test_y = np.argmax(y_test, axis=1)

print(classification_report(test_y, pred, labels=list(range(len(labels)))))


# In[48]:


# Download image with bottle from custom data
load_img('Bouteille.jpg')


# In[17]:


# Extract names of the class labels
labels_names = []
for i in range(len(labels)):
    labels_names += [i]
    
reverse_mapping = dict(zip(labels_names, labels)) 

def mapper(value):
    return reverse_mapping[value]


# In[25]:


# Pre-process image 1
image_1 = load_img('Bouteille.jpg', target_size=(32, 32))
image_1 = img_to_array(image_1) 
image_1 = image_1 / 255.0
prediction_image_1 = np.array(image_1)
prediction_image_1 = np.expand_dims(image_1, axis=0)


# In[26]:


# Get prediction 1
prediction_1 = model.predict(prediction_image_1)
value_1 = np.argmax(prediction_1)
name_1 = mapper(value_1)
print(f'Prediction is {name_1}.')


# In[45]:


# Download second image with bicycle from custom data
load_img('images.jpg')


# In[46]:


# Pre-process image 2
image_2 = load_img('images.jpg', target_size=(32, 32))
image_2 = img_to_array(image_2) 
image_2 = image_2 / 255.0
prediction_image_2 = np.array(image_2)
prediction_image_2 = np.expand_dims(image_2, axis=0)


# In[47]:


# Get prediction 2
prediction_2 = model.predict(prediction_image_2)
value_2 = np.argmax(prediction_2)
name_2 = mapper(value_2)
print(f'Prediction is {name_2}.')


# In[ ]:





# In[ ]:


# References
# 1> CNN Tensorflow -> https://www.tensorflow.org/tutorials/images/cnn
# 2> CIFER-100 -> https://www.kaggle.com/datasets/fedesoriano/cifar100?select=train
# 3> https://www.kaggle.com/code/faressayah/cifar-10-images-classification-using-cnns-88/notebook#%F0%9F%94%84-Data-Preprocessing

