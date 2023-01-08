#!/usr/bin/env python
# coding: utf-8

# In[36]:


import os
import warnings
warnings.simplefilter('ignore')


# In[37]:


import numpy as np
import pandas as pd


# In[38]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[39]:


from skimage.io import imread,imshow
from skimage.transform import resize
from skimage.color import rgb2gray


# In[44]:


jen=os.listdir("C:/Image recognition/jen")


# In[45]:


manasa=os.listdir("C:/Image recognition/manasa")


# In[46]:


ipsh=os.listdir("C:/Image recognition/ipsh")


# In[47]:


limit=10

manasa_images=[None]*limit
j=0
for i in manasa:
    if(j<limit):
        manasa_images[j]=imread("C:/Image recognition/manasa/"+i)
        j+=1
    else:
        break


# In[48]:


limit=10

jen_images=[None]*limit
j=0
for i in jen:
    if(j<limit):
        jen_images[j]=imread("C:/Image recognition/jen/"+i)
        j+=1
    else:
        break


# In[49]:


limit=10

ipsh_images=[None]*limit
j=0
for i in ipsh:
    if(j<limit):
        ipsh_images[j]=imread("C:/Image recognition/ipsh/"+i)
        j+=1
    else:
        break


# In[50]:


imshow(jen_images[0])


# In[51]:


imshow(jen_images[3])


# In[52]:


imshow(manasa_images[7])


# In[53]:


imshow(ipsh_images[0])


# In[54]:


jen_gray=[None]*limit
j=0

for i in jen:
    if(j<limit):
        jen_gray[j]=rgb2gray(jen_images[j])
        j+=1
    else:
        break


# In[55]:


manasa_gray=[None]*limit
j=0

for i in manasa:
    if(j<limit):
        manasa_gray[j]=rgb2gray(manasa_images[j])
        j+=1
    else:
        break


# In[56]:


ipsh_gray=[None]*limit
j=0

for i in ipsh:
    if(j<limit):
        ipsh_gray[j]=rgb2gray(ipsh_images[j])
        j+=1
    else:
        break


# In[57]:


imshow(jen_gray[7])


# In[58]:


imshow(manasa_gray[9])


# In[59]:


imshow(ipsh_gray[6])


# In[60]:


jen_gray[3].shape


# In[64]:


for j in range (10):
    js=jen_gray[j]
    jen_gray[j]=resize(js,(512,512))


# In[65]:


for j in range (10):
    mc=manasa_gray[j]
    manasa_gray[j]=resize(mc,(512,512))


# In[66]:


for j in range (10):
    ip=ipsh_gray[j]
    ipsh_gray[j]=resize(ip,(512,512))


# In[67]:


imshow(manasa_gray[5])


# In[68]:


manasa_gray[1].shape
ipsh_gray[4].shape


# # Jennifer Sharon
# 

# In[69]:


len_of_images_jen=len(jen_gray)


# In[70]:


len_of_images_jen


# In[71]:


image_size_jen=jen_gray[1].shape


# In[72]:


image_size_jen


# In[73]:


flatten_size_jen=image_size_jen[0]*image_size_jen[1]


# In[74]:


flatten_size_jen


# In[75]:


for i in range(len_of_images_jen):
    jen_gray[i]=np.ndarray.flatten(jen_gray[i].reshape(flatten_size_jen,1))


# In[76]:


jen_gray=np.dstack(jen_gray)


# In[77]:


jen_gray.shape


# In[78]:


jen_gray=np.rollaxis(jen_gray,axis=2,start=1)


# In[79]:


jen_gray=jen_gray.reshape(len_of_images_jen,flatten_size_jen)


# In[80]:


jen_gray.shape


# In[81]:


jen_data=pd.DataFrame(jen_gray)


# In[82]:


jen_data


# In[83]:


jen_data["label"]="Jennifer Sharon"


# In[84]:


jen_data


# # Manasa Chowdary 
# 

# In[85]:


len_of_images_manasa=len(manasa_gray)


# In[86]:


len_of_images_manasa


# In[87]:


image_size_manasa=manasa_gray[1].shape


# In[88]:


image_size_manasa


# In[89]:


flatten_size_manasa=image_size_manasa[0]*image_size_manasa[1]


# In[90]:


flatten_size_manasa


# In[91]:


for i in range(len_of_images_manasa):
    manasa_gray[i]=np.ndarray.flatten(manasa_gray[i].reshape(flatten_size_manasa,1))


# In[92]:


manasa_gray=np.dstack(manasa_gray)


# In[93]:


manasa_gray.shape


# In[94]:


manasa_gray=np.rollaxis(manasa_gray,axis=2,start=1)


# In[95]:


manasa_gray=manasa_gray.reshape(len_of_images_manasa,flatten_size_manasa)


# In[96]:


manasa_gray.shape


# In[97]:


manasa_data=pd.DataFrame(manasa_gray)


# In[98]:


manasa_data


# In[102]:


manasa_data["label"]="Manasa"


# In[103]:


manasa_data


# # Ipshita Balaji 
# 

# In[104]:


len_of_images_ipsh=len(ipsh_gray)


# In[105]:


len_of_images_ipsh


# In[106]:


image_size_ipsh=ipsh_gray[1].shape


# In[107]:


image_size_ipsh


# In[108]:


flatten_size_ipsh=image_size_ipsh[0]*image_size_ipsh[1]


# In[109]:


flatten_size_ipsh


# In[110]:


for i in range(len_of_images_ipsh):
    ipsh_gray[i]=np.ndarray.flatten(ipsh_gray[i].reshape(flatten_size_ipsh,1))


# In[111]:


ipsh_gray=np.dstack(ipsh_gray)


# In[112]:


ipsh_gray.shape


# In[113]:


ipsh_gray=np.rollaxis(ipsh_gray,axis=2,start=1)


# In[114]:


ipsh_gray=ipsh_gray.reshape(len_of_images_ipsh,flatten_size_ipsh)


# In[115]:


ipsh_gray.shape


# In[116]:


ipsh_data=pd.DataFrame(ipsh_gray)


# In[117]:


ipsh_data


# In[118]:


ipsh_data["label"]="Ipshita"


# In[119]:


ipsh_data


# # Concatenation

# In[144]:


friend_1=pd.concat([ipsh_data,manasa_data])


# In[145]:


friend=pd.concat([friend_1,jen_data])


# In[146]:


friend


# # Reshuffle
# 

# In[534]:


from sklearn.utils import shuffle


# In[535]:


girl_indexed=shuffle(friend).reset_index()


# In[536]:


girl_indexed


# In[537]:


girl=girl_indexed.drop(['index'],axis=1)


# In[538]:


girl


# In[539]:


x=girl.values[:,:-1]


# In[540]:


y=girl.values[:,-1]


# In[541]:


x


# In[542]:


y


# # Assigning Training and Test Dataset

# In[543]:


from sklearn.model_selection import train_test_split


# In[544]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# # SVM Algorithm

# In[545]:


from sklearn import svm


# In[546]:


clf=svm.SVC()
clf.fit(x_train,y_train)


# # Image Prediction

# In[547]:


y_pred=clf.predict(x_test)


# In[548]:


y_pred


# In[549]:


for i in (np.random.randint(0,6,4)):
    predicted_images=(np.reshape(x_test[i],(512,512)).astype(np.float64))
    plt.title('Predicted Label: {0}'. format(y_pred[i]))
    plt.imshow(predicted_images,interpolation='nearest',cmap='gray')
    plt.show()


# # Prediction Accuracy

# In[550]:


from sklearn import metrics


# In[551]:


accuracy=metrics.accuracy_score(y_test,y_pred)


# In[552]:


accuracy


# # Error Analysis of Prediction

# In[553]:


from sklearn.metrics import confusion_matrix


# In[554]:


confusion_matrix(y_test,y_pred)


# In[ ]:




