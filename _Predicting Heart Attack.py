#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[21]:


data = pd.read_csv('heart.csv')
data


# In[22]:


data.isnull().sum()


# In[23]:


data.describe()


# In[24]:


data.info()


# In[25]:


data.duplicated().sum()


# In[26]:


data.drop_duplicates(inplace=True)


# In[27]:


data.duplicated().sum()


# In[28]:


sns.histplot(x="age",data=data);


# In[29]:


s=data["sex"].value_counts().reset_index()
px.pie(s,names="index",values="sex",title="%AGE OF MALE AND FEMALE PATIENTS:")


# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


x=data.drop("output",axis=1).values
y=data["output"].values
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.5)


# In[32]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[33]:


from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(x_train, y_train)


# In[34]:


reg.score(x_train,y_train)


# In[35]:


from xgboost import XGBClassifier
from sklearn.metrics import r2_score

xgb = XGBClassifier(colsample_bylevel= 0.9,
                    colsample_bytree = 0.8, 
                    gamma=0.99,
                    max_depth= 5,
                    min_child_weight= 1,
                    n_estimators= 8,
                    nthread= 5,
                    random_state= 0,
                    )
xgb.fit(x_train,y_train)


# In[36]:


print('Accuracy of XGBoost classifier on training set: {:.2f}'
     .format(xgb.score(x_train, y_train)))
print('Accuracy of XGBoost classifier on test set: {:.2f}'
     .format(xgb.score(x_test, y_test)))


# In[37]:


from sklearn.metrics import accuracy_score


# In[38]:


y_pred=xgb.predict(x_test)
print("Accuracy of XG Boost model is:",
accuracy_score(y_test, y_pred)*100)


# In[ ]:





# In[ ]:




