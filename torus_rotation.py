import numpy as np
import matplotlib.pyplot as plt
import math 
import random


r2 = np.sqrt(2)
a = math.pi #the parameter \alpha
bs = [i*r2 for i in range(1,5)] #the parameters \beta we sweep over
N = int(1e6) #length of trajectory simulation
Nsample = int(1e5) #number of random samples for visualizing invariant measures
fig, ax = plt.subplots(2, 4, figsize=(18, 10), subplot_kw={'projection': '3d'},dpi = 300)
for i in range(4):
    #define the torus map for given parameters
    b = bs[i]
    print(r'Testing parameters (a,b) = {}'.format([a,b]))

    def tmap(x,y):
        return (a+x)%1,(b+y)%1
  
    #simulate long trajectory
    X = np.zeros((N,2))
    x,y = .1,.1
    for j in range(N):
        X[j,:] = x,y
        x,y = tmap(x,y)
        
    #embed in Euclidean space using toroidal coordinates
    R = 3
    r = 1
    Y = np.zeros((N,3))
    Y[:,0] = (R+r*np.cos(2*math.pi*X[:,0]))*np.cos(2*math.pi*X[:,1])
    Y[:,1] = (R+r*np.cos(2*math.pi*X[:,0]))*np.sin(2*math.pi*X[:,1])
    Y[:,2] = r*np.sin(2*math.pi*X[:,0])
    
    
    #form delay-coordinate trajectory
    X_delay = np.zeros((N-2,3))
    X_delay[:,0] = Y[2:,0]
    X_delay[:,1] = Y[1:-1,0]
    X_delay[:,2] = Y[:-2,0]
    
    
    #random sample for visualization
    ixs = random.sample(range(0,len(Y)-2),Nsample)
    ixs2 = random.sample(range(0,len(Y)-2),Nsample)
    Y = Y[ixs]
    X_delay = X_delay[ixs2]
    
    #make plots    
    ax[0,i].scatter(Y[:,0], Y[:,1],Y[:,2], c='k', s = .0005)
    ax[0,i].view_init(elev=35, azim=40)
    ax[0, i].set_title(f'Original: $(\\alpha, \\beta) = (\\pi, {i+1} \\sqrt{{2}})$', fontsize=15)
    ax[0,i].set_xlabel('x',fontsize =15)
    ax[0,i].set_ylabel('y',fontsize =15)
    ax[0,i].set_zlabel('z',fontsize =15)
    ax[0,i].set_xlim(-4,4)
    ax[0,i].set_ylim(-4,4)
    ax[0,i].set_zlim(-2,2)
    ax[0,i].grid(False)
    ax[1,i].scatter(X_delay[:,0], X_delay[:,1], X_delay[:,2], c='k', s = .0005)
    ax[1,i].view_init(elev=35, azim=40)
    ax[1, i].set_title(f'Delay: $(\\alpha, \\beta) = (\\pi, {i+1} \\sqrt{{2}})$', fontsize=15)
    ax[1,i].set_xlim(-4,4)
    ax[1,i].set_ylim(-4,4)
    ax[1,i].set_zlim(-4,4)
    ax[1,i].set_xlabel('$x_k$',fontsize =15)
    ax[1,i].set_ylabel('$x_{k-1}$',fontsize =15)
    ax[1,i].set_zlabel('$x_{k-2}$',fontsize =15)
    ax[1,i].grid(False)
    
plt.savefig('Torus_example')