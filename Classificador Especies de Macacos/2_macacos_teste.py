
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
from keras.models import model_from_json
from keras.preprocessing import image


# In[3]:


arquivo = open('model_json.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()


# In[6]:


model = model_from_json(estrutura_rede)
model.load_weights('model_macacos.h5')


# In[91]:


result = []

test_image = image.load_img('database/teste/erythrocebus_patas.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

pred = model.predict_on_batch(test_image)
result.append(pred)

result = np.asarray(result)
imprime = np.array(result[0][0])
#print(imprime)


# In[92]:


if imprime[0] == 1:
    print('Alouatta palliata')
elif imprime[1] == 1:
    print('Erythrocebus patas')
elif imprime[2] == 1:
    print('Cacajao calvus')
elif imprime[3] == 1:
    print('Macaco-japonês')
elif imprime[4] == 1:
    print('Cebuella pygmaea')
elif imprime[5] == 1:
    print('Cebus capucinus')
elif imprime[6] == 1:
    print('Mico argentatus')
elif imprime[7] == 1:
    print('saimiri_sciureus')
elif imprime[8] == 1:
    print('Aotus nigriceps')
elif imprime[9] == 1:
    print('Trachypithecus johnii')
else:
    print('Especia não classificada')

