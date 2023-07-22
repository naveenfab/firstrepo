#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
import os 
#for dirname, _,filename in os.walk():
 #   for filename in filenmaes:
  #      print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt


# In[36]:


df=pd.read_csv('Desktop/project/HR-Employee-Attrition.csv')


# In[37]:


df.head(10)


# In[38]:


df.shape


# In[39]:


df.info()


# In[40]:


df.describe()


# In[41]:


df.isnull().sum()


# In[42]:


attrition_count =pd.DataFrame(df['Attrition'].value_counts())
attrition_count


# In[43]:


plt.pie(attrition_count['Attrition'] , labels=['No' , 'Yes'], explode=(0.2,0))


# In[44]:


sns.countplot(df['Attrition'])


# In[45]:


df.drop(['EmployeeCount', 'EmployeeNumber'],axis = 1)


# In[46]:


attrition_dummies = pd.get_dummies(df['Attrition'])
attrition_dummies.head()


# In[47]:


df=pd.concat([df, attrition_dummies] , axis=1)
df.head()


# In[48]:


df.drop(['No'], axis =1)


# In[49]:


sns.barplot(x='Gender', y='Yes', data =df)


# In[50]:


sns.countplot(df['EducationField'])
fig=plt.gcf()
fig.set_size_inches(24,24)
plt.title('EducationField')


# In[51]:


sns.barplot(x='Department',y='Yes',data=df)


# In[52]:


sns.barplot(x='BusinessTravel', y = 'Yes', data =df)


# In[66]:


sns.countplot(df['Age'])
fig=plt.gcf()
fig.set_size_inches(24,24)
plt.title('Age')


# In[53]:


sns.countplot(df['DistanceFromHome'])
fig=plt.gcf()
fig.set_size_inches(24,24)
plt.title('DistanceFromHome')


# # Modelling

# In[54]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr())


# # Data Preprocessing

# In[55]:


from sklearn.preprocessing import LabelEncoder
for column in df.columns:
    if df[column].dtype==np.number:
        continue
    else:
            df[column]=LabelEncoder().fit_transform(df[column])


# # Model Building

# In[56]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators =10, criterion ='entropy', random_state =0)


# In[57]:


x= df.drop(['Yes'], axis =1)
y = df['Yes']


# In[58]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size =0.3,random_state=0)


# In[59]:


x_train.head()


# In[60]:


rf.fit(x_train, y_train)


# In[61]:


rf.score(x_train, y_train)


# In[62]:


pred = rf.predict(x_test)


# In[63]:


from sklearn.metrics import accuracy_score


# In[64]:


accuracy_score(y_test, pred)


# In[ ]:




