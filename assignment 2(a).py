#!/usr/bin/env python
# coding: utf-8

# In[1]:


# IMPORTING THE REQUIRED LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# READING THE CSV FILE 
df=pd.read_csv(r"C:\Users\amitt\OneDrive\Desktop\A2Q2Data_train.csv",header=None)


# In[3]:


# CALCULATING THE X MATRIX AND Y MATRIX
y=df[100]
df.drop(df.columns[[100]], axis=1, inplace=True)
data=df.to_numpy()
label=y.to_numpy().reshape(-1,1)
ones_column = np.ones(data.shape[0]).reshape((-1,1))
data = np.concatenate((data, ones_column), axis = 1)


# In[4]:


# CALCULATING THE REQUIRED W(ML)
transposedata=np.transpose(data)
mul=np.matmul(transposedata, data)
inversemat=np.linalg.inv(mul)
x_mul_y=np.matmul(transposedata, label)
W_ml=np.matmul(inversemat, x_mul_y)
print("the W(ml) is below")
print(W_ml)


# In[5]:


wtx=np.matmul(data, W_ml)


# In[6]:


# plotting the diagram for Actual nd predicted label 
for i in range(10000):
    plt.scatter(label[i][0], wtx[i][0])
titletemp="plotting between actual label vs predicted label" 
plt.title(titletemp)
plt.xlabel("Actual label")
plt.ylabel("predicted label")
plt.show


# In[ ]:




