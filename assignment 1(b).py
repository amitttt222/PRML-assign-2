#!/usr/bin/env python
# coding: utf-8

# In[57]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math


# In[58]:


#reading inputs
df=pd.read_csv(r"C:\Users\amitt\OneDrive\Desktop\A2Q1.csv",header=None)
array=df.to_numpy()


# In[59]:


# initalizing few parameters
k=4
k_point=[4,1,2,10]
miu_arr=array[k_point]
covariance_matrix=np.cov(array.T,bias=True)
covariance_matrix.shape
covariance_arr=[covariance_matrix]*k
pi_arr=[0.25]*k

rows, cols = (array.shape[0],k)
lemda_arr = [[0]*cols]*rows       #lemda array

d=array.shape[1]


# In[60]:


# calculation of initial lemdas
def pk_calculation(i,k):
    det_k=np.linalg.det(covariance_arr[k])
#     if(det_k==0):
#         return 0.3
    det_k=det_k**.5
    temp=(2*math.pi)**(d/2)
    dum=temp*det_k
    first_term=1/(dum)
    second_term=np.subtract(array[i].T,miu_arr[k])
    cov_inv=np.linalg. pinv(covariance_arr[k])
    temp=np.matmul((second_term.T),cov_inv)
    second_term=np.matmul(temp,second_term)
    
    return second_term
    second_term=np.multiply(second_term,-0.5)
#     second_term=math.exp(second_term)
    second_term=math.exp(10)
    output=first_term*second_term
    output*=pi_arr[k]
    return output

    
print(pk_calculation(0,0))


# In[61]:


# calculation of lemdas
def lemda_cal():
    for i in range(array.shape[0]):
        for j in range(k):
            numer=pk_calculation(i,j)
            denom=0
            for l in range(k):
                denom+=pk_calculation(i,l)
            if(denom==0):
                denom=0.01
            lemda_arr[i][j]=numer/denom

        


# In[62]:


# calculation of first log likelihood
def calculate_log_likelihood():
    log_val=0
    for i in range(array.shape[0]):
        temp=0
        for j in range(k):
            temp+=pk_calculation(i,j)
        log_val+=math.log(abs(temp))
    return log_val
            


# In[56]:


lemda_cal()
log_val=calculate_log_likelihood
iteration=[0]
log_val_list=[log_val]

it=0

while(it<100):
    it+=1
    # calculation of miu
    for i in range (k):
        numer=np.zeros(array.shape[1])
        denom=0
        for j in range(array.shape[0]):
            numer=np.add(numer,np.multiply(array[j],lemda_arr[j][i]))
            denom+=lemda_arr[j][i]
        #if(denom==0):
        #    demom=0.1
        miu_arr[i]=np.multiply(numer,(1/denom))
    
    #calculation of sigma that is covariance matrix
    for i in range(k):
        numer=np.zeros( (array.shape[1],array.shape[1]) , dtype=np.int64)
        denom=0
        for j in range(array.shape[0]):
            temp=array[j].T
            temp2=miu_arr[i]
            temp3=np.subtract(temp,temp2)
            temp4=temp3.T
            temp5=np.matmul(temp3,temp4)
            temp5=np.multiply(temp5,lemda_arr[j][i])
            numer=np.add(numer,temp5)
            denom+=lemda_arr[j][i]
        covariance_arr[i]=np.multiply(numer,(1/denom))
        
    # calculation of pi
    for i in range(k):
        numer=0
        for j in range(array.shape[0]):
            numer+=lemda_arr[j][i]
        pi_arr[i]=numer/array.shape[0]
    
    # calculations of lemdas
    lemda_cal()
    
    l=calculate_log_likelihood()
    if(it==5 ):
    #if(it==10):
        log_val_list.append(l)
        iteration.append(it)
        break
    else:
        log_val_list.append(l)
        iteration.append(it)
            
iteration = np.array(iteration)
log_val_list=np.array(log_val_list)
plt.figure(1)
titletemp="LOG LIKELIHOOD OF Ith ITERATION"
plt.title(titletemp)
plt.xlabel("Iteration")
plt.ylabel("value")
plt.plot(iteration,log_val_list)

print("value at each iterations are")
for i in range(len(log_val_list)):
    print(i,log_val_list[i])
                
        

            
    


# In[ ]:





# In[ ]:




