import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from scipy.stats import norm
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import make_grid as make_image_grid
from tqdm import tnrange

torch.manual_seed(2017) # reproducability
sns.set_style('dark')
#matplotlib inline

# Model
class VAE(nn.Module):
    def __init__(self,latent_dim=20,hidden_dim=500):#20,hidden_dim=500):
        super(VAE,self).__init__()
        self.fc_e = nn.Linear(784,hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim,latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim,latent_dim)
        self.fc_d1 = nn.Linear(latent_dim,hidden_dim)
        self.fc_d2 = nn.Linear(hidden_dim,784)
            
    def encoder(self,x_in):
        x = F.relu(self.fc_e(x_in.view(-1,784)))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar
    
    def decoder(self,z):
        z = F.relu(self.fc_d1(z))
        x_out = F.sigmoid(self.fc_d2(z))
        return x_out.view(-1,1,28,28)
    
    def sample_normal(self,mean,logvar):
        # Using torch.normal(means,sds) returns a stochastic tensor which we cannot backpropogate through.
        # Instead we utilize the 'reparameterization trick'.
        # http://stats.stackexchange.com/a/205336
        # http://dpkingma.com/wordpress/wp-content/uploads/2015/12/talk_nips_workshop_2015.pdf
        std = torch.exp(logvar*0.5)#std = torch.exp(0.5 * logvar)
        eps = torch.distributions.normal.Normal(torch.zeros(std.size()),torch.ones(std.size()), validate_args=None)
        eps = eps.sample()
        z = eps.mul(std).add_(mean)#mean + std * eps
        #e = Variable(torch.randn(sd.size())) # Sample from standard normal
        #z = e.mul(sd).add_(mean)
        return z
    
    def forward(self,x_in):
        z_mean, z_logvar = self.encoder(x_in)
        z = self.sample_normal(z_mean,z_logvar)
        x_out = self.decoder(z)
        return x_out, z_mean, z_logvar

model = VAE()

# Loss function
def criterion(x_out,x_in,z_mu,z_logvar):
    bce_loss = F.binary_cross_entropy(x_out,x_in,size_average=False)
    kld_loss = -0.5 * torch.sum(1 + z_logvar - (z_mu ** 2) - torch.exp(z_logvar))  
    
    #kl_values = -0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    loss = (bce_loss + kld_loss) / x_out.size(0) # normalize by batch size # 原来的代码
    #kl_means = torch.mean(kl_values, dim=0)
    #loss = torch.sum(kl_means)
    return loss, kld_loss/ x_out.size(0), bce_loss/ x_out.size(0)   # 修改的代码

# Optimizer
optimizer = torch.optim.Adam(model.parameters())

# Data loaders
trainloader = DataLoader(
    MNIST(root='./data',train=True,download=True,transform=transforms.ToTensor()),
    batch_size=128,shuffle=True)
testloader = DataLoader(
    MNIST(root='./data',train=False,download=True,transform=transforms.ToTensor()),
    batch_size=128,shuffle=True)

# Training
def train(model,optimizer,dataloader,epochs=15):
    losses = []
    kld_lossesN20 = []
    for epoch in tnrange(epochs,desc='Epochs'):
        for images,_ in dataloader:
            x_in = Variable(images)
            optimizer.zero_grad()
            x_out, z_mu, z_logvar = model(x_in)
            loss, kld_loss, bce_loss  = criterion(x_out,x_in,z_mu,z_logvar)  # 修改的代码
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            kld_lossesN20.append(kld_loss.item())
        print('{}/{}, loss: {}, kld_loss: {}, bce_loss: {}'.format(epoch+1, epochs, loss.item(), kld_loss.item(), bce_loss.item()))
    return losses, kld_lossesN20

train_losses, train_kld_lossesN20 = train(model,optimizer,trainloader)
plt.figure(figsize=(10,5))
plt.plot(train_losses)
plt.title("Zhuyonggui1")
plt.show()

#a = mu
filename = 'data1/train_kld_lossesN20.npy'
np.save(filename, train_kld_lossesN20)
kld_lossesDataN20 = np.load(filename)
print('kld_lossesDataN20.shape = ', kld_lossesDataN20.shape)
kld_lossesDataN20Mean = kld_lossesDataN20.mean()
print('kld_lossesDataN20Mean = ', kld_lossesDataN20Mean)

