import matplotlib.pyplot as plt
import numpy as np
from geomloss import SamplesLoss
import torch.nn as nn
import torch
from torch import optim
from torchdiffeq import odeint_adjoint as odeint
from sklearn.preprocessing import MaxAbsScaler
import pickle


with open("IM_recon.p", "rb") as f:
    data1 = pickle.load(f)

y, xs_IM,L1 = data1[0], data1[2], data1[3]
with open("DIM_recon.p", "rb") as f:
    data2 = pickle.load(f)

y, xs_DIM,L2 = data2[0], data2[2], data2[3]
yix = data1[1]
N = 500
L1mid = np.convolve(L1, np.ones(N)/N, mode='valid')
L2mid = np.convolve(L2, np.ones(N)/N, mode='valid')
# iters = [i for i in range(len(L1))]
# iters_conv = [i+250 for i in range(len(L1)-500)]
y = y[::100]
length = int(1e3)
start = int(0)
fig,ax = plt.subplots(2,2,figsize = (15,10),dpi = 300)
lw = .5
ms = 2
iters = [i for i in range(len(L1))]
iters2 = [i for i in range(len(L1mid))]

ax[0,0].plot(y[:,0][start:start+length],y[:,2][start:start+length],linewidth = lw,linestyle = '-',marker = 'o',markersize = ms,color = 'dimgrey')  
ax[1,0].plot(xs_IM[:,0][start:start+length],xs_IM[:,2][start:start+length],linewidth = lw,linestyle = '-',marker = 'o',markersize = ms,color = 'darkred')  
ax[0,1].plot(iters,L1,color = 'darkred',linewidth = 2,alpha = .3,label = r'$\mathcal{J}_1$')
ax[0,1].plot(iters2[::100],L1mid[::100],linestyle = '--',color = 'darkred',label = r'$\mathcal{J}_1$ (avg.)')

ax[1,1].plot(xs_DIM[:,0][start:start+length],xs_DIM[:,2][start:start+length],linewidth = lw,linestyle = '-',marker = 'o',markersize = ms,color = 'royalblue')  

ax[0,1].plot(iters,np.array(L2),color = 'royalblue',linewidth = 2,alpha = .3,label = r'$\mathcal{J}_2$')
ax[0,1].plot(iters2[::100],np.array(L2mid[::100]),linestyle = '--',color = 'royalblue',label = r'$\mathcal{J}_2$ (avg.)')
ax[0, 1].legend(loc='best',fontsize = 15)  # Add legend to J1 subplot
ax[0,1].set_yscale('log')

ax[0,0].set_title('Ground Truth',fontsize = 18)
ax[1,0].set_title(r'Reconstruction with $\mathcal{J}_1(\theta)$',fontsize = 18)
ax[1,1].set_title(r'Reconstruction with $\mathcal{J}_2(\theta)$',fontsize = 18)
ax[0,0].set_title('Ground Truth',fontsize = 18)
ax[0,1].set_title('Training Loss',fontsize = 18)


ax[0,0].set_xlabel(r'$x$',fontsize = 18)
ax[0,0].set_ylabel(r'$z$',fontsize = 18)

ax[1,1].set_xlabel(r'$x$',fontsize = 18)
ax[1,1].set_ylabel(r'$z$',fontsize = 18)

ax[1,0].set_xlabel(r'$x$',fontsize = 18)
ax[1,0].set_ylabel(r'$z$',fontsize = 18)

ax[0,1].set_xlabel(r'Iterations',fontsize = 18)
plt.subplots_adjust(hspace = .3)

ax[0,0].set_xlim(-20,20)
ax[0,0].set_ylim(0,45)
ax[1,0].set_xlim(-20,20)
ax[1,0].set_ylim(0,45)
ax[1,1].set_xlim(-20,20)
ax[1,1].set_ylim(0,45)