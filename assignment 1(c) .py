#!/usr/bin/env python
# coding: utf-8

# In[135]:


# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math


# In[136]:


# reading data
df=pd.read_csv(r"C:\Users\amitt\OneDrive\Desktop\A2Q1.csv",header=None)
array=df.to_numpy()


# In[137]:


# initial initialization
def initial_assignment(k):
    assign=np.zeros([array.shape[0]])
    import random
    for i in range(array.shape[0]):
        assign[i]=random.randrange(k)
    seta=set(assign)
    while(len(seta)!=k):
        for i in range(array.shape[0]):
            assign[i]=random.randrange(k)
    return assign


# In[138]:


# finding initial mean
def find_initial_mean(k,assign):
    mean=np.zeros([k,array.shape[1]])
    count=np.zeros([k])
    for i in range(array.shape[0]):
        mean[int(assign[i])]= np.add(mean[int(assign[i])],array[i])
        count[int(assign[i])]+=1
    for i in range(len(mean)):
        mean[i]=np.divide(mean[i], count[i])
    return mean


# In[139]:


# implementation of k-mean algorithm
def k_mean_algo(k,assign,mean):
    it=0
    final_error = -1
    itlist=[]
    errorlist=[]
    while(1):
        it+=1
        itlist.append(it)
        dist=0
        error=0
        for i in range(array.shape[0]):
            dist=np.linalg.norm(array[i] - mean[int(assign[i])])
            dist*=dist
            error+=dist
        errorlist.append(error)
        error_fig=plt
        #print(error)
        if(final_error == -1):
            final_error = error
        elif(final_error > error):
            final_error = error
        else:
            break
        reassign=np.zeros(array.shape[0])
        dist_arr=[0]*k
        for i in range(array.shape[0]):
            for j in range(k):
                dist_arr[j]=np.linalg.norm(array[i] - mean[j])
                dist_arr[j]*=dist_arr[j]
            minpos=dist_arr.index(min(dist_arr))
            if(assign[i]==minpos):
                reassign[i]=assign[i]
            else:
                reassign[i]=minpos
        assign=reassign
        mean=np.zeros([k,array.shape[1]])
        count=np.zeros([k])  
        for i in range(array.shape[0]):
            mean[int(assign[i])]=mean[int(assign[i])]+array[i]
            count[int(assign[i])]+=1

        for i in range(len(mean)):
            mean[i]=np.divide(mean[i], count[i])

    returnlist=[]
    returnlist.append(itlist)
    returnlist.append(errorlist)
    returnlist.append(assign)
    returnlist.append(mean)
    return returnlist

    


# In[140]:


# plotting of objective with respect to iterations
def objective_plot(i,itlist,errorlist):
    if(i==0):
        i="FIRST"
    if(i==1):
        i="SECOND"
    if(i==2):
        i="THIRD"
    if(i==3):
        i="FOURTH"
    if(i==4):
        i="FIFTH"
    itlist = np.array(itlist)
    errorlist=np.array(errorlist)
    plt.figure(1)
    titletemp="OBJECTIVE CONVERGENCE DIAGRAM" # FOR" + " " + i + " " "RANDOM INITIALIZATION "
    plt.title(titletemp)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.plot(itlist,errorlist)
    
    


# In[141]:


# calling different functions and getting values
def starting(i):
    k=4
    assign=initial_assignment(k)
    mean=find_initial_mean(k,assign)
    returnlist=k_mean_algo(k,assign,mean)
    itlist=returnlist[0]
    errorlist=returnlist[1]
    assign=returnlist[2]
    mean=returnlist[3]
    objective_plot(i,itlist,errorlist)
    print("objective value is",errorlist[-1])
        
    print("each data going to cluster number are below")
    for i in range(array.shape[0]):
        print(i+1," ",assign[i])
        
# starting of code
for i in range(1):
    starting(i)


# In[ ]:





# In[ ]:




