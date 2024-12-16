import matplotlib.pyplot as plt
import numpy as np
from geomloss import SamplesLoss
import torch.nn as nn
import torch
from torch import optim
from sklearn.preprocessing import MaxAbsScaler
import pickle

############################# lorenz system
def lorenz(xyz, *, s=10, r=28, b=2.667):
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])

################################ experiment parameters
dt = 0.001 #time-step
num_steps = int(1e6) #length of trajectory simulation
y = np.zeros((num_steps + 1, 3))  
y[0] = np.array([-5.065457, -7.56735 , 19.060379])  # Set initial values
tau = 100 #time-delay
method = 'IM' #select either 'delay_IM' or 'IM' to choose between a delay-coordinate invariant measure loss or a state-coordinate invariant measure loss
num_samples = 2000 #the number of trajectory samples we want to learn from. The runtime is very sensitive to this number since we do not use minibatches. For > 2000, one should partition into mini-batches. 
Nsteps = 30000 #how many training iterations to take 
plot_every = 500 #how often to plot training result

###############################################

#simulate trajectory
for i in range(num_steps):
    y[i + 1] = y[i] + lorenz(y[i]) * dt

#rescale for NN learning
transformer = MaxAbsScaler().fit(y)
y = transformer.transform(y)

#time-delay map
def delay(X):
    dim = 6
    new = np.zeros((len(X)-tau*dim,dim))
    for i in range(dim):
        print(dim*tau-(i+1)*tau)
        new[:,i] = X[dim*tau-(i+1)*tau:-(1+i)*tau]
    return new

#form delayed trajectory
y_delay = delay(y[:,0])
Ty_true = y[tau:tau+len(y_delay)]
y = y[:len(y_delay)]

#take random samples for training
import random
batch_ixs = list(range(0,len(y)))
ixs = random.sample(batch_ixs,num_samples)

############################## Build network 
torch.manual_seed(1293452) #random seed so that initialization is controlled for comparison

net = nn.Sequential(
    nn.Linear(3, 100),
    nn.Tanh(),
    nn.Linear(100,100),
    nn.Tanh(),
    nn.Linear(100,100),
    nn.Tanh(),
    nn.Linear(100,3))

############################## Define Loss
loss = SamplesLoss(loss="energy")
optimizer = optim.Adam(net.parameters(), lr=1e-3)
  
##############################  Training loop
net.train()
losses = []
for step in range(Nsteps):
    y_batch = torch.tensor(y[ixs],dtype = torch.float,requires_grad = True)     #\mu_*
    Ty_batch = torch.tensor(Ty_true[ixs],dtype = torch.float,requires_grad = True)  #T_*\#\mu_*
    y_delay_batch = torch.tensor(y_delay[ixs],dtype = torch.float,requires_grad = True) #\Psi_*\#\mu_*
    optimizer.zero_grad()      
    Ty = net(y_batch) #T_{\theta}\#\mu_*
    TTy = net(Ty)
    TTTy = net(TTy)
    TTTTy = net(TTTy)
    TTTTTy = net(TTTTy)
    DIM = torch.cat((TTTTTy[:,0].unsqueeze(0),TTTTy[:,0].unsqueeze(0),TTTy[:,0].unsqueeze(0),TTy[:,0].unsqueeze(0),Ty[:,0].unsqueeze(0),y_batch[:,0].unsqueeze(0)),dim = 0).T #\Psi_{\theta}\#\mu_*
    #delay invariant measure
    if method =='delay_IM':
        L = loss(Ty,Ty_batch) + loss(DIM,y_delay_batch)
    if method == 'IM':
        L = loss(Ty,Ty_batch)
    losses.append(L.detach().numpy())
    
    print('iteration ', step, 'loss ', L)
    L.backward()
    optimizer.step()
    if step%plot_every == 0:
        # state coordinate push forward
        plt.title('State-Coordinate Pushforward Samples',fontsize = 15)
        plt.scatter(Ty.detach().numpy()[:,0],Ty.detach().numpy()[:,1],s =1)
        plt.scatter(Ty_batch.detach().numpy()[:,0],Ty_batch.detach().numpy()[:,1], s =1)
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.show()
        #delay coordinate push forward
        plt.title('Delay-Coordinate Pushforward Samples',fontsize = 15)
        plt.scatter(DIM.detach().numpy()[:,0],DIM.detach().numpy()[:,1],s =1)
        plt.scatter(y_delay_batch.detach().numpy()[:,0],y_delay_batch.detach().numpy()[:,1], s =1)
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.show()
        #traj simulation
        x = y_batch[0]
        xs = []
        for i in range(int(1e4)):
            xs.append(x.detach().numpy())
            x  = net(x)
        xs = transformer.inverse_transform(np.array(xs))
        plt.scatter(xs[:,0],xs[:,2],s = .1)
        plt.show()


#Simulate a long trajectory using the learned model        
net.eval()
x = torch.tensor(y[0],dtype = torch.float)
xs = []

for i in range(int(1e4)):
    xs.append(x.detach().numpy())
    x  = net(x)
xs = transformer.inverse_transform(np.array(xs))
y = transformer.inverse_transform(y)
plt.scatter(xs[:,0],xs[:,2],s = .1)
plt.show() 
        
plt.plot(xs[:,0][:5000][5:],xs[:,2][:5000][5:],linewidth = .5,linestyle = '--',marker = 'o',markersize = 3) 

if method == 'delay_IM': 
    with open("DIM_recon.p", "wb") as f:
        pickle.dump([y,y[ixs],xs,losses], f)
if method == 'IM':
    with open("IM_recon.p", "wb") as f:
        pickle.dump([y,y[ixs],xs,losses], f)

plt.plot(losses)
plt.yscale('log')
