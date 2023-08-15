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


# CALCULATING GRADIENT DESCENT AND REQUIRED ALGORITHM
def gradientdescent(mul,x_mul_y,W0,W_ml,eta):
    Wt=W0
    x_mul_y=np.multiply(x_mul_y,2)
    i=1
    errorvec=[]
    itlist=[]
    import math
    print("||Wt − Wml|| at each iteration is given below")
    while(i<1000):
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
        print(l2n)
        errorvec.append(l2n)
        i+=1
        
    # PLOTTING THE REQUIRED GRAPH  
    itlist = np.array(itlist)
    errorlist=np.array(errorvec)
    plt.figure(1)
    titletemp="CONVERGENCE DIAGRAM FOR W" 
    plt.title(titletemp)
    plt.xlabel("Iteration")
    plt.ylabel("||Wt − Wml||")
    plt.plot(itlist,errorlist)


# In[4]:


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

# INITIALIZATION OF INITIAL W
W0 = np.zeros((100, 1))
import random
for i in range(100):
    x = random.random()
    W0[i][0]=x
eta=0.1
# CALLING THE gradientdescent FUNCTION
gradientdescent(mul,x_mul_y,W0,W_ml,eta)


# In[ ]:




