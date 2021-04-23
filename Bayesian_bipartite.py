#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial


# In[2]:


def calcQ(q,ntot1,ntot2):
    #print("Ntot",ntot1,ntot2)
    ### q [0:ntot1+1,0:ntot2+1]
    mat_b = q
    fac = np.zeros(ntot1+2)
    facinv = np.zeros(ntot1+2)
    deriv_b = np.zeros((ntot1+2,ntot2+2))
    deriv_d = np.zeros((ntot1+2,ntot2+2))
    deriv_x = np.zeros((ntot1+2,ntot2+2))
    mat_a = np.zeros((ntot1+2,ntot2+2))
    mat_x = np.zeros((ntot1+2,ntot2+2))
    mat_y = np.zeros((ntot1+2,ntot2+2))
    mat_m = np.zeros((ntot1+2,ntot2+2))
    
    
    fac[0] = 1.0
    fac[1] = 1.0
    facinv[0] = 1.0
    facinv[1] = 1.0
    
    for i in range(2,ntot1+1):
        fac[i] = fac[i-1]*i
        facinv[i] = 1.0/fac[i]
    
    for i in range(0,ntot1+1):
        for j in range(0,ntot2+1):
            #print(i,j)
            deriv_b[i,j]= mat_b[i+1,j]*(i+1)
    for i in range(0,ntot1+1):
        for j in range(0,ntot2+1):
            deriv_d[i,j]= mat_b[i,j+1]*(j+1)
            
    mat_a[0,0] = mat_b[0,0]        
    for j in range(0,ntot2+1):
        deriv_x[0,j] = mat_b[0,j+1]*(j+1)
        
    mat_a[0,1] = deriv_x[0,0]
    mat_x = deriv_x
    for j in range(2,ntot2+1):
        mat_y = calc2(ntot1,ntot2,deriv_d,mat_x)
        mat_a[0,j] = mat_y[0,0] * facinv[j]
        mat_x = np.zeros((ntot1+2,ntot2+2))
        mat_x = mat_y
    
    mat_m = np.zeros((ntot1+2,ntot2+2))
    for j in range(0,ntot2+1):
        for i in range(0,ntot1+1):
            mat_m[i,j] = mat_b[i+1,j]*(i+1)
    
    mat_a[1,0] = mat_m[0,0]
    mat_x = np.zeros((ntot1+2,ntot2+2))
    mat_x = mat_m
    for j in range(0,ntot2+1):
        mat_y = calc2(ntot1,ntot2,deriv_d,mat_x)
        mat_a[1,j]=mat_y[0,0]*facinv[j]
        mat_x = np.zeros((ntot1+2,ntot2+2))
        mat_x=mat_y

    
    for i in range(2,ntot1+1):
        mat_x = np.zeros((ntot1+2,ntot2+2))
        #print("mat_m",mat_m)
        
        mat_x = calc1(ntot1,ntot2,deriv_b,mat_m)
        mat_a[i,0]=mat_x[0,0]*facinv[i]  # Corrected error here 2017.02.10
        mat_c = mat_x
        for j in range(1,ntot2+1):
            mat_y = np.zeros((ntot1+2,ntot2+2))
            mat_y = calc2(ntot1,ntot2,deriv_d,mat_x)
            mat_a[i,j]=mat_y[0,0]*facinv[i]*facinv[j]
            mat_x = np.zeros((ntot1+2,ntot2+2))
            mat_x=mat_y
        
        mat_m=mat_c
        mat_c = np.zeros((ntot1+2,ntot2+2))

    Qn = mat_y[0,0]*facinv[ntot1]*facinv[ntot2]

    
    C = np.zeros((ntot1+2,ntot2+2))
    nk = np.zeros((ntot1+2,ntot2+2))
    #Calculate <n_i,j>
    for i in range(0,ntot1+1):
        for j in range(0,ntot2+1):
            k1=ntot1-i
            k2=ntot2-j
            C[k1,k2]=mat_b[k1,k2]*mat_a[i,j]
            nk[k1,k2]=C[k1,k2]/Qn
    return (Qn,nk)


def calc1(ntot1,ntot2,deriv_b,mat_m):

    mat_x = np.zeros((ntot1+2,ntot2+2))
    mat_u = mat_m
    deriv_u = np.zeros((ntot1+2,ntot2+2))
    mat_v = np.zeros((ntot1*2+2,ntot2*2+2))
    mat_w = np.zeros((ntot1+2,ntot2+2))
    i = 0
    for m in range(0,ntot2+1):
        for j in range(0,ntot1+1):
            i += 1
            deriv_u[j,m]=mat_u[j+1,m]*(j+1)


    for j in range(0,ntot1+1):
        for p in range(0,ntot1+1):
            for l in range(0,ntot2+1):
                for k in range(0,ntot2+1):
                    r = j+p
                    s = l+k
                    if r < (ntot1+2) and s < (ntot2+2) :
                        mat_v[r,s]=mat_v[r,s]+mat_u[j,l]*deriv_b[p,k]

    
    for l in range(0,ntot1+1):
        for m in range(0,ntot2+1):
            mat_w[l,m]=deriv_u[l,m]+mat_v[l,m]      
    return mat_w

def calc2(ntot1,ntot2,deriv_d,mat_n):
    deriv_o = np.zeros((ntot1+2,ntot2+2))
    mat_p = np.zeros((ntot1*2+2,ntot2*2+2))
    mat_q = np.zeros((ntot1+2,ntot2+2))
    for j in range(0,ntot2+1):
        deriv_o[0,j]=mat_n[0,j+1]*(j+1)
    for l in range(0,ntot2+1):
        for k in range(0,ntot2+1):
            s=l+k
            mat_p[0,s]=mat_p[0,s]+mat_n[0,l]*deriv_d[0,k]
    for l in range(0,ntot2+1):
        mat_q[0,l]=deriv_o[0,l]+mat_p[0,l]
        
    return mat_q
            
            


# In[ ]:





# ## N = 20

# In[139]:


### MATRIX format: q_init, n_sim_, nk (n_model)
### List of values showing up: q_init_input, n_sim_input, nk_
### only q values : q_init_vec (including 0,1 and 1,0 )

#### make a vector with predefined numbers and q_init
input_N = 20
### q_init_input is all the numbers showing in simulation
### 
q_init_input = pd.read_csv("q_init_input_{0}.txt".format(input_N),sep='\t',header=None)
q_init_input.columns=["j","k","q"]
q_init_input["lnq"] = np.log(q_init_input["q"])
#q_init_input.drop(["q"],axis=1)
#q_init_vec = q_init_input.values[:,-1]
### q_init is a matrix
N = input_N + 2


q_init = np.zeros((N,N))
q_init[0,0] = 1
q_init[0,1] = 1
q_init[1,0] = 1
for i in range(q_init_input.shape[0]):
    q_init[q_init_input.loc[i,"j"],q_init_input.loc[i,"k"]]= q_init_input.loc[i,"q"]
#print("q_init",q_init)
np.savetxt("qinit_20_matrix.txt",q_init)


lnq_init = np.zeros((N,N))   
for i in range(q_init_input.shape[0]):
    lnq_init[q_init_input.loc[i,"j"],q_init_input.loc[i,"k"]]= q_init_input.loc[i,"lnq"]
#print("lnq_init",lnq_init)


n_sim_input = pd.read_csv("n_sim_input_{0}.txt".format(input_N),sep='\t',header=None)
n_sim_input.columns=["j","k","nsim"]

### n_sim_ is a matrix
n_sim = np.zeros((N,N))
for i in range(n_sim_input.shape[0]):
    n_sim[n_sim_input.loc[i,"j"],n_sim_input.loc[i,"k"]]= n_sim_input.loc[i,"nsim"]
n_sim
np.savetxt("nsim_20_matrix.txt",n_sim)



Q,nk = calcQ(q_init,N-2,N-2)
#nk




# In[140]:


f = -np.log(Q)+ sum(sum(n_sim*lnq_init))
grad = n_sim-nk

#print("grad",grad)
#print("x0",x0)

f_ = [f]
##################################
k = 0
x0 = lnq_init
alpha = 0.1
while k < 100:
    x = (x0 + alpha*grad) * (x0 != 0)
    #print("x",x)
    
    ### revert x to qmatrix 
    q = np.exp(x0) * (x0 != 0)
    q[0,0] = 1
    q[1,0] = 1
    q[0,1] = 1
    Q,nk = calcQ(q,N-2,N-2)
    f = -np.log(Q) + sum(sum(n_sim*x))
    print("Q,f",Q,f)
    f_.append(f)
    
    #### update
    grad = n_sim-nk
    x0 = x
    
    k += 1

plt.plot(f_)


# In[136]:


np.savetxt("qout_20.txt",q)


# In[137]:


np.savetxt("nkout_20.txt",nk)



### error analysis
#(n_sim - nk)/n_sim




# MSE
sum(sum((n_sim - nk)**2))


# In[155]:


### check on the total
nk_sum = np.zeros((N,N))
for i in range(nk.shape[0]):
    for j in range(nk.shape[1]):
        nk_sum[i,j] = nk[i,j] * j
sum(sum(nk_sum))


# In[157]:


## Output nk as vectors
# nk_ = []
# for i in range(q_init_input.shape[0]):
#     nk_.append(nk[q_init_input.loc[i,"j"],q_init_input.loc[i,"k"]])
# nk_ = np.array(nk_)
# np.savetxt("nkout_20_vector.txt",nk_)

nk_ = []
for i in range(nk.shape[0]):
    for j in range(nk.shape[1]):
        if nk[i,j] != 0:
            nk_.append([i,j,nk[i,j]])
        
np.savetxt("nkout_20_all.txt",nk_)


# In[ ]:


### Calculate convergence


# ## N = 20 & 30 Multitraj

# In[64]:


### MATRIX format: q_init, n_sim_, nk (n_model)
### List of values showing up: q_init_input, n_sim_input, nk_
### only q values : q_init_vec (including 0,1 and 1,0 )

#### make a vector with predefined numbers and q_init
input_N = 30
trajN = [20,30]
### q_init_input is all the numbers showing in simulation
### 
q_init_input = pd.read_csv("q_init_input_{0}.txt".format(input_N),sep='\t',header=None)
q_init_input.columns=["j","k","q"]
q_init_input["lnq"] = np.log(q_init_input["q"])
#q_init_input.drop(["q"],axis=1)
#q_init_vec = q_init_input.values[:,-1]
### q_init is a matrix
N = input_N + 2



q_init = np.zeros((N,N))
q_init[0,0] = 1
q_init[0,1] = 1
q_init[1,0] = 1
for i in range(q_init_input.shape[0]):
    q_init[q_init_input.loc[i,"j"],q_init_input.loc[i,"k"]]= q_init_input.loc[i,"q"]
#print("q_init",q_init)

lnq_init = np.zeros((N,N))   
for i in range(q_init_input.shape[0]):
    lnq_init[q_init_input.loc[i,"j"],q_init_input.loc[i,"k"]]= q_init_input.loc[i,"lnq"]
#print("lnq_init",lnq_init)


f_sum = 0
grad_sum = 0
grad_sum_ = 0
n_sim_input = {}
n_sim = {}
#######################################################################################
for N_traj in trajN:
    n_sim_input[N_traj] = pd.read_csv("n_sim_input_{0}.txt".format(N_traj),sep='\t',header=None)
    n_sim_input[N_traj].columns=["j","k","nsim"]
    N_traj_tot = N_traj+2
    
    ### n_sim_ is a matrix
    n_sim[N_traj] = np.zeros((N_traj_tot,N_traj_tot))
    for i in range(n_sim_input[N_traj].shape[0]):
        n_sim[N_traj][n_sim_input[N_traj].loc[i,"j"],n_sim_input[N_traj].loc[i,"k"]]= n_sim_input[N_traj].loc[i,"nsim"]
    #n_sim
    
    
    ### Generate objective function values
    Q,nk = calcQ(q_init[:N_traj_tot,:N_traj_tot],N_traj_tot-2,N_traj_tot-2)
    #print("Q",Q,n_sim,lnq_init)
    f = -np.log(Q)+ sum(sum(n_sim[N_traj]*lnq_init[:N_traj_tot,:N_traj_tot]))
    print("f",f)
    f_sum = f_sum + f
    print("f_sum",f_sum)
    
    ### Generate grad values
    grad = n_sim[N_traj]-nk
    if type(grad_sum) != int:
        grad_sum_ = np.zeros((N_traj_tot,N_traj_tot))
        for i in range(grad_sum.shape[0]):
            for j in range(grad_sum.shape[1]):
                grad_sum_[i,j] = grad_sum[i,j]
    print("N",N_traj)
    grad_sum = grad_sum_ + grad
    
#######################################################################################


# In[65]:


#print("grad",grad)
#print("x0",x0)

f_ = [f_sum]
##################################
k = 0
x0 = lnq_init
alpha = 2
while k < 1:
    x = (x0 + alpha*grad_sum) * (x0 != 0)
    #print("x",x)
    
    ### revert x to qmatrix 
    q = np.exp(x0) * (x0 != 0)
    q[0,0] = 1
    q[1,0] = 1
    q[0,1] = 1
    
    f_sum = 0 
    grad_sum = 0
    grad_sum_ = 0
    for N_traj in trajN:
        N_traj_tot = N_traj+2
        Q,nk = calcQ(q[:N_traj_tot,:N_traj_tot],N_traj_tot-2,N_traj_tot-2)
        f = -np.log(Q) + sum(sum(n_sim[N_traj]*x[:N_traj_tot,:N_traj_tot]))
        print("Q,f",Q,f)
        f_sum = f_sum + f
        
        ### grad
        grad = n_sim[N_traj]-nk
        if type(grad_sum) != int:
            grad_sum_ = np.zeros((N_traj_tot,N_traj_tot))
            for i in range(grad_sum.shape[0]):
                for j in range(grad_sum.shape[1]):
                    grad_sum_[i,j] = grad_sum[i,j]
        grad_sum = grad_sum_ + grad
        
        
    f_.append(f_sum)
    print("f_sum",f_sum)
    #### update
    x0 = x
    k += 1

plt.plot(f_)



### N = 20 & 30


# In[134]:


### MATRIX format: q_init, n_sim_, nk (n_model)
### List of values showing up: q_init_input, n_sim_input, nk_
### only q values : q_init_vec (including 0,1 and 1,0 )

#### make a vector with predefined numbers and q_init
input_N = 20
### q_init_input is all the numbers showing in simulation
### 
q_init_input = pd.read_csv("q_init_input_{0}.txt".format(input_N),sep='\t',header=None)
q_init_input.columns=["j","k","q"]
q_init_input["lnq"] = np.log(q_init_input["q"])
#q_init_input.drop(["q"],axis=1)
#q_init_vec = q_init_input.values[:,-1]
### q_init is a matrix
N = input_N + 2


q_init = np.zeros((N,N))
q_init[0,0] = 1
q_init[0,1] = 1
q_init[1,0] = 1
for i in range(q_init_input.shape[0]):
    q_init[q_init_input.loc[i,"j"],q_init_input.loc[i,"k"]]= q_init_input.loc[i,"q"]
#print("q_init",q_init)

lnq_init = np.zeros((N,N))   
for i in range(q_init_input.shape[0]):
    lnq_init[q_init_input.loc[i,"j"],q_init_input.loc[i,"k"]]= q_init_input.loc[i,"lnq"]
#print("lnq_init",lnq_init)


n_sim_input = pd.read_csv("n_sim_input_{0}.txt".format(input_N),sep='\t',header=None)
n_sim_input.columns=["j","k","nsim"]

### n_sim_ is a matrix
n_sim = np.zeros((N,N))
for i in range(n_sim_input.shape[0]):
    n_sim[n_sim_input.loc[i,"j"],n_sim_input.loc[i,"k"]]= n_sim_input.loc[i,"nsim"]
n_sim




Q,nk = calcQ(q_init,N-2,N-2)
#nk


f = -np.log(Q)+ sum(sum(n_sim*lnq_init))
grad = n_sim-nk

#print("grad",grad)
#print("x0",x0)

f_ = [f]
##################################
k = 0
x0 = lnq_init
alpha = 0.1
while k < 100:
    x = (x0 + alpha*grad) * (x0 != 0)
    #print("x",x)
    
    ### revert x to qmatrix 
    q = np.exp(x0) * (x0 != 0)
    q[0,0] = 1
    q[1,0] = 1
    q[0,1] = 1
    Q,nk = calcQ(q,N-2,N-2)
    f = -np.log(Q) + sum(sum(n_sim*x))
    print("Q,f",Q,f)
    f_.append(f)
    
    #### update
    grad = n_sim-nk
    x0 = x
    
    k += 1

plt.plot(f_)




x = np.random.randn(2)
k = 1
tol = 0.0001


def grad(x): return np.array([8*x[0]-4*x[1], -4*x[0]+4*x[1]])


def f(x): return 4*x[0]**2+2*x[1]**2-4*x[0]*x[1]


conv = grad(x)@grad(x)
fs = [f(x)]
convL = [conv]
while conv > tol:
    x = x - 0.01*grad(x)
    fs.append(f(x))
    conv = grad(x)@grad(x)
    convL.append(conv)
    k = k+1

print(x)
plt.figure()
plt.plot(fs)
plt.xlabel('Iterations')
plt.ylabel('Function Value')
plt.figure()
plt.plot(convL)
plt.xlabel('Iterations')
plt.ylabel('Convergence Criterion')


# In[ ]:




