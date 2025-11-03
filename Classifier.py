#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K


# In[2]:


# root = '/content/drive/MyDrive/Ravaee/GTEX/'


# In[3]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


x_train =np.load(root+'data.npy')
y_train=np.load(root+'y.npy')
num_classes = y_train.shape[1]


# In[6]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)


# In[7]:


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5 ,restore_best_weights=True)


# In[8]:


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[9]:


model = Sequential()

model.add(Conv2D(32 , kernel_size=15 , strides=2 , padding="same" , input_shape=(128,128,1)))
model.add(LeakyReLU(alpha=0.2))

          
model.add(Conv2D(256 , kernel_size=15 , strides=2 , padding="same" ))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.5))

model.add(Conv2D(512 , kernel_size=15 , strides=2 , padding="same" ))
model.add(LeakyReLU(alpha=0.2))

          
model.add(Conv2D(768 , kernel_size=15 , strides=2 , padding="same" ))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dropout(0.2))

# softmax classifier
model.add(Dense(num_classes))
model.add(Activation("softmax"))


# In[10]:


opt = Adam(learning_rate=1e-4)

model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy",precision_m,recall_m,f1_m] )
model.summary()


# In[11]:


hist = model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), callbacks=[stop_early])


# In[12]:


import matplotlib.pyplot as plt


# In[13]:


plt.plot(hist.epoch , hist.history['loss'])
plt.plot(hist.epoch , hist.history['val_loss'])
plt.legend(['loss','val_loss'])
plt.show()


# In[14]:


plt.plot(hist.epoch , hist.history['accuracy'])
plt.plot(hist.epoch , hist.history['val_accuracy'])
plt.legend(['accuracy','val_accuracy'])
plt.show()


# In[16]:


model.evaluate(x_train, y_train)


# In[17]:


model.evaluate(x_val, y_val)


# In[15]:


# model.save(root+'classifier.h5')

