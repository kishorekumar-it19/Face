#!/usr/bin/env python
# coding: utf-8

# # Building Machine Learning Model

# # Part1-Data Preprocessing

# Step1- Importing Libraries for Preprocessing

# In[64]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# 
# Step2- Import Dataset

# In[65]:


dataset =pd.read_csv('finalpro.csv')


# In[66]:


type(dataset)


# In[67]:


dataset


# In[77]:


dataset.iloc[:,-2]


# In[68]:


dataset.head()


# In[69]:


dataset.head(10)


# Step3- Split Independent and Dpendent Variables

# In[70]:


x= dataset.iloc[:,:2]


# In[71]:


x


# In[72]:


x.iloc[:,-1]


# In[73]:


def convert_to_int(word):
    word_dict = {'E':1, 'W':2, 'N':3, 'S':4, 'NE':5, 'NW':6, 'SE':7, 'SW':8
                }
    return word_dict[word]


# In[74]:


x.iloc[:,-1] = x.iloc[:,-1].apply(lambda a:convert_to_int(a))


# In[75]:


x


# In[ ]:


def convert_to_int(word):
    word_dict = {'E':1, 'W':2, 'N':3, 'S':4, 'NE':5, 'NW':6, 'SE':7, 'SW':8
                }
    return word_dict[word]


# In[78]:


dataset.iloc[:,-2] = dataset.iloc[:,-2].apply(lambda a:convert_to_int(a))


# In[79]:


dataset


# In[76]:


type(x)


# In[93]:


x= dataset.iloc[:,:2].values #convert from dataframe to numpy array


# In[81]:


x


# In[82]:


y= dataset.iloc[:,2:].values


# In[12]:


y


# In[83]:


x.ndim #mandatory to be in 2 dimesion for Linear Regression


# In[84]:


type(x)


# In[85]:


dataset.corr()


# In[19]:





# In[86]:





# In[87]:


x


# In[17]:




# In[16]:





# In[88]:


x


# In[ ]:





# In[89]:


y


# In[90]:


x#removing the column which is not required .so removing first column


# In[91]:





# In[94]:


x


# In[95]:


import seaborn as sns
sns.heatmap(dataset.corr(),annot=True)


# In[96]:


dataset.hist()


# In[97]:


y


# In[31]:


#y= dataset.iloc[:,1:].values


# In[32]:


#y


# If you have null values in the dataset 

# Step6- Split Test and Train Data

# In[99]:


from sklearn.model_selection import train_test_split                #previously cros_validation was used in sklearn
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)


# In[100]:


x_train


# In[101]:


x_test


# In[ ]:





# 

# In[102]:


from sklearn.linear_model import LinearRegression


# In[103]:


lr=LinearRegression()


# In[104]:


lr.fit(x_train,y_train)


# In[105]:


y_predict=lr.predict(x_test)


# In[106]:


y_predict


# In[111]:


y_train


# In[112]:


y_predict.mean()


# In[110]:


print(lr.predict(np.array([[4,2]])))


#In[43]:
pickle.dump(lr, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))




# In[44]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




