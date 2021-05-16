#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[11]:


df = pd.read_csv("carprices.csv")
df


# In[12]:


plt.scatter(df['Mileage'],df['Sell Price($)'])


# In[13]:


plt.scatter(df['Age(yrs)'],df['Sell Price($)'])


# In[15]:


X = df[['Mileage','Age(yrs)']]
y = df['Sell Price($)']


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)


# In[36]:


clf = LinearRegression()
clf.fit(X_train, y_train)


# In[37]:


clf.predict(X_test)


# In[38]:


clf.score(X_test, y_test)

