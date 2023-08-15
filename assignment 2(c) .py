#!/usr/bin/env python
# coding: utf-8

# In[125]:


# IMPORTING THE REQUIRED LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[126]:


# READING THE CSV FILE 
df=pd.read_csv(r"C:\Users\amitt\OneDrive\Desktop\A2Q2Data_train.csv",header=None)


# In[127]:


# CALCULATING GRADIENT DESCENT AND REQUIRED ALGORITHM
def gradientdescent(mul,x_mul_y,W0,eta):
    Wt=W0
    W_ml=W0
    x_mul_y=np.multiply(x_mul_y,2)
    i=1
    errorvec=[]
    itlist=[]
    import math
    while(i<2000):
        #eta=1/i
        itlist.append(i)
        t1=np.matmul(mul, Wt)
        t1=np.multiply(t1,2)
        grad = np.subtract(t1, x_mul_y)
        div = np.linalg.norm(grad)
        div=1/div
        grad=np.multiply(grad,div)
        grad=np.multiply(grad,eta)
        W_next=np.subtract(Wt, grad)
        Wt=W_next
        error=np.subtract(W_ml,W_next)
        l2n = np.linalg.norm(error)
        #print(l2n)
        errorvec.append(l2n)
        i+=1
    return Wt  


# In[128]:


# CALCULATING THE X MATRIX AND Y MATRIX
y=df[100]
df.drop(df.columns[[100]], axis=1, inplace=True)
data=df.to_numpy()
label=y.to_numpy().reshape(-1,1)

# CALCULATING THE REQUIRED W(ML)
transposedata=np.transpose(data)
mul=np.matmul(transposedata, data)
inversemat=np.linalg.inv(mul)
x_mul_y=np.matmul(transposedata, label)
W_ml=np.matmul(inversemat, x_mul_y)

import random
W0 = np.zeros((100, 1))
import random
for i in range(100):
    x = random.random()
    W0[i][0]=x
eta=0.1
vec=[]
itlist=[]
for i in range(100):
    itlist.append(i)

    randomlist=[];
    for j in range(100):
        num1 = random.randint(0, 9999)
        randomlist.append(num1)
    randomlist=np.array(randomlist)
    samplex=data[randomlist]
    sampley=label[randomlist]

    transposedata=np.transpose(samplex)
    mul=np.matmul(transposedata, samplex)
    x_mul_y=np.matmul(transposedata, sampley)
    
    Wt=gradientdescent(mul,x_mul_y,W0,eta)
    W0=Wt
    temp=W_ml-Wt
    l2n = np.linalg.norm(temp)
    vec.append(l2n)

itlist=np.array(itlist) 
vec=np.array(vec)
plt.figure(1)
titletemp="DIAGRAM OF ||Wt-Wml||2 AT EACH ITERATION" 
plt.title(titletemp)
plt.xlabel("Iteration")
plt.ylabel("||Wt-Wml||2")
plt.plot(itlist,vec)
    

