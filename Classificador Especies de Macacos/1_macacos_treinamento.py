
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


# In[12]:


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())


# In[13]:


model.add(Dense(units = 32, activation = 'relu'))
model.add(Dense(units = 10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[14]:


data_train = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7, 
                                         horizontal_flip = True,
                                         shear_range=0.2,
                                         height_shift_range=0.07,
                                         zoom_range=0.2)

data_test = ImageDataGenerator(rescale = 1./255)


# In[15]:


base_train = data_train.flow_from_directory('database/training_set',
                                            target_size = (64, 64),
                                            batch_size = 10,
                                            class_mode = 'categorical')

base_test = data_test.flow_from_directory('database/test_set',
                                          target_size = (64, 64),
                                          batch_size = 10,
                                          class_mode = 'categorical')


# In[16]:


model.fit_generator(base_train,
                    steps_per_epoch = 16,
                    epochs = 100,
                    validation_data = base_test,
                    validation_steps = 4)


# In[22]:


result = []

test_image = image.load_img('database/test_set/n9/n910.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

pred = model.predict_on_batch(test_image)
result.append(pred)

result = np.asarray(result)
result


# In[26]:


model_json = model.to_json()


# In[27]:


with open('model_json.json', 'w') as json_file:
    json_file.write(model_json)


# In[29]:


model.save_weights('model_macacos.h5')

