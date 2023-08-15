#!/usr/bin/env python
# coding: utf-8

# In[167]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math


# In[168]:


#reading data
df=pd.read_csv(r"C:\Users\amitt\OneDrive\Desktop\A2Q1.csv",header=None)
array=df.to_numpy()


# In[169]:


#calculate log likelihood 
def calculate_log_likelihood(pi,p,c,d):
    import math
    log_val=0
    for i in range(array.shape[0]):
        temp_val=0
        for j in range(k):
            temp=pi[j]*(p[j]**c[i])*((1-p[j])**(d-c[i]))
            temp_val+=temp
        #print(i," ",temp_val)
        log_val+=math.log(abs(temp_val))
        #log_val+=(temp_val)
    return log_val

k=4   # number of mixture
pi=[]    # pi array
norm1=0
for i in range(k):
    x=random.randint(1,10)
    norm1+=x
    pi.append(x)
for i in range(k):
    pi[i]=pi[i]/norm1

p=[]        # p array  
for i in range(k):
    p.append(random.random())

rows, cols = (array.shape[0],k)
lemda_arr = [[0]*cols]*rows       #lemda array

c=[0]*array.shape[0]              
for i in range(array.shape[0]):
    for j in range(array.shape[1]):
        if(array[i][j]==1):
            c[i]+=1
            
d=array.shape[1]


# calculation of initial lemdas------------------------------

for i in range(array.shape[0]):
    for j in range(k):
        numerator=pi[j]*(p[j]**c[i])*(1-p[j]**(d-c[i]))
        denominator=0
        for g in range(k):
            denominator+=pi[g]*(p[g]**c[i])*(1-p[g]**(d-c[i]))        
        lemda_arr[i][j]=numerator/denominator
        
#-------------------------------------------------------------


log_val=calculate_log_likelihood(pi,p,c,d)

it=0
iteration=[0]
log_val_list=[log_val]


while(it<100):
    #temp_lemda_arr=[[0]*]
    it+=1
    #calculate lemda
    for i in range(array.shape[0]):
        for j in range(k):
            numerator=pi[j]*(p[j]**c[i])*(1-p[j]**(d-c[i]))
            denominator=0
            for g in range(k):
                denominator+=pi[g]*(p[g]**c[i])*(1-p[g]**(d-c[i]))
                    
            lemda_arr[i][j]=numerator/denominator

    
    #calculate pi, p
    for j in range(k):
        temp_lemda=0
        for i in range(array.shape[0]):
            temp_lemda+=lemda_arr[i][j]
        temp_lemda=temp_lemda/array.shape[0]
        pi[j]=temp_lemda
    
    for j in range(k):
        numer=0
        denom=array.shape[0]
        for i in range(array.shape[0]):
            numer+=lemda_arr[i][j]*c[i]
        p[j]=numer/denom
    
    l=calculate_log_likelihood(pi,p,c,d)

    if(it==50 or abs(log_val_list[-1]-l)<1e-15):
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
    
#print(log_val_list)

            
            
    

