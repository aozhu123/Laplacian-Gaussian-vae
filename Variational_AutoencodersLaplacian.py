import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.distributions.laplace import Laplace  # 新增加的代码

from scipy.stats import norm
from scipy import stats  # 新增加的代码
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import make_grid as make_image_grid
from tqdm import tnrange
# 与Variational_AutoencodersLaplace相比除了print("test_loss = ", test_loss)增加了print("test_kld_loss = ", test_kld_loss) print("test_bce_loss = ", test_bce_loss) 

torch.manual_seed(2017) # reproducability
sns.set_style('dark')
#matplotlib inline

# Model
class VAE(nn.Module):
    def __init__(self,latent_dim=20,hidden_dim=500):#20,hidden_dim=500):
        super(VAE,self).__init__()
        self.fc_e = nn.Linear(784,hidden_dim)
        self.fc_alpha = nn.Linear(hidden_dim,latent_dim)
        self.fc_beta = nn.Linear(hidden_dim,latent_dim)
        self.fc_d1 = nn.Linear(latent_dim,hidden_dim)
        self.fc_d2 = nn.Linear(hidden_dim,784)
            
    def encoder(self,x_in):
        x = F.relu(self.fc_e(x_in.view(-1,784)))
        #mean = self.fc_mean(x)  # 原来的代码
        #logvar = self.fc_logvar(x)   # 原来的代码
        #return mean, logvar  # 原来的代码
        alpha = self.fc_alpha(x)    # 新增加的代码
        logbeta = self.fc_beta(x)    # 新增加的代码
        return alpha, logbeta   # 新增加的代码
        
        
        
    
    def decoder(self,z):
        z = F.relu(self.fc_d1(z))
        x_out = F.sigmoid(self.fc_d2(z))
        return x_out.view(-1,1,28,28)
    
    """def sample_normal(self,mean,logvar):  # 原来的代码
        # Using torch.normal(means,sds) returns a stochastic tensor which we cannot backpropogate through.
        # Instead we utilize the 'reparameterization trick'.
        # http://stats.stackexchange.com/a/205336
        # http://dpkingma.com/wordpress/wp-content/uploads/2015/12/talk_nips_workshop_2015.pdf
        sd = torch.exp(logvar*0.5)
        e = Variable(torch.randn(sd.size())) # Sample from standard normal
        z = e.mul(sd).add_(mean)
        return z"""   # 原来的代码

    def sample_laplace(self,alpha,logvar):  # 新增加的代码
        # Using torch.normal(means,sds) returns a stochastic tensor which we cannot backpropogate through.
        # Instead we utilize the 'reparameterization trick'.
        # http://stats.stackexchange.com/a/205336
        # http://dpkingma.com/wordpress/wp-content/uploads/2015/12/talk_nips_workshop_2015.pdf
        beta = torch.div(torch.exp(logvar*0.5), 2**(0.5))#torch.exp(logvar)# 新增加的代码
        eps =torch.distributions.laplace.Laplace(torch.zeros(beta.size()),torch.ones(beta.size()), validate_args=None)
        eps = eps.sample()
        #q_distribution = Laplace(0,1)  #  Sample from standard laplace  # 新增加的代码
        #h1 = q_distribution.sample()
        #e = Variable(h1)
        z = eps.mul(beta).add_(alpha) #z = eps.mul(std).add_(mean) # 新增加的代码 重参数化
        return z     # 新增加的代码
    
    """def forward(self,x_in):  # 原来的代码
        z_mean, z_logvar = self.encoder(x_in)
        z = self.sample_normal(z_mean,z_logvar)
        x_out = self.decoder(z)
        return x_out, z_mean, z_logvar"""   # 原来的代码

    def forward(self,x_in):   # 新增加的代码
        z_alpha, z_logbeta = self.encoder(x_in)  # 新增加的代码
        z = self.sample_laplace(z_alpha,z_logbeta) # 新增加的代码
        x_out = self.decoder(z)    # 新增加的代码
        return x_out, z_alpha, z_logbeta   # 新增加的代码

model = VAE()

"""# Loss function  # 原来的代码
def criterion(x_out,x_in,z_mu,z_logvar):
    bce_loss = F.binary_cross_entropy(x_out,x_in,size_average=False)
    kld_loss = -0.5 * torch.sum(1 + z_logvar - (z_mu ** 2) - torch.exp(z_logvar))
    loss = (bce_loss + kld_loss) / x_out.size(0) # normalize by batch size
    return loss"""   # 原来的代码

# Loss function
def criterion(x_out,x_in,z_alpha,z_logbeta):  # 新增加的代码
    #crit=nn.L1Loss()
    bce_loss = F.binary_cross_entropy(x_out,x_in,size_average=False)  # crit(x_out, x_in)#criteon=nn.L1Loss()
    # 下面是KL散度 第二种方法 这种方法是正确的
    beta = torch.div(torch.exp(z_logbeta*0.5), 2**(0.5))#torch.exp(z_logbeta)
    #logbeta = z_logbeta#torch.div(torch.exp(z_logbeta*0.5), 2**(0.5))# 新增加的代码
    alpha = z_alpha
    #n = len(alpha)
    p1 = -torch.sum(torch.log(beta) + 1)#-torch.sum(logbeta + 1)#
    p2 = torch.sum(torch.mul(beta,torch.exp(-torch.div(torch.abs(alpha),beta))))
    p3 = torch.sum(torch.abs(alpha))
    kld_loss = p1 + p2 +p3
    #L1loss = torch.abs(x_out).sum()
    #L2loss = torch.norm(x_out)
    loss = (bce_loss + kld_loss)/ x_out.size(0)# + L2loss ) / x_out.size(0)#+L1loss) / x_out.size(0)
    #kld_loss = -0.5 * torch.sum(1 + z_logvar - (z_mu ** 2) - torch.exp(z_logvar))
    #loss = (bce_loss + kld_loss) / x_out.size(0) # normalize by batch size
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
    kld_losses20 = []
    for epoch in tnrange(epochs,desc='Epochs'):
        for images,_ in dataloader:
            x_in = Variable(images)
            optimizer.zero_grad()
            x_out, z_alpha, z_logbeta = model(x_in)#x_out, z_mu, z_logvar = model(x_in)# 原来的代码
            loss, kld_loss, bce_loss = criterion(x_out,x_in,z_alpha,z_logbeta) # 修改的代码
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            kld_losses20.append(kld_loss.item())
        print('{}/{}, loss: {}, kld_loss: {}, bce_loss: {}'.format(epoch+1, epochs, loss.item(), kld_loss.item(), bce_loss.item()))
    return losses, kld_losses20

train_losses, train_kld_losses20 = train(model,optimizer,trainloader)
plt.figure(figsize=(10,5))
plt.plot(train_losses)
plt.title("Zhuyonggui1")
plt.show()

#a = mu
filename = 'data1/train_kld_losses20.npy'
np.save(filename, train_kld_losses20)
kld_lossesData20 = np.load(filename)
print('kld_lossesData20.shape = ', kld_lossesData20.shape)
kld_lossesData20Mean = kld_lossesData20.mean()
print('kld_lossesData20Mean = ', kld_lossesData20Mean)




