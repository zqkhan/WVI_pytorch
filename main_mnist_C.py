from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils.helpers import compute_cost
from torch.distributions import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import time
from utils.sinkhorn import SinkhornSolver

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print (torch.cuda.is_available())
torch.manual_seed(args.seed)
dataset_dimension = 28


device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
datasets.MNIST('../data', train=True, download=True,
transform=transforms.Compose([
    transforms.Resize((dataset_dimension,dataset_dimension)),transforms.ToTensor(),
                   ])),batch_size=args.batch_size, shuffle=True, ** kwargs)

class A_Encoder(nn.Module):
    def __init__(self,image_size=784,latent_dim=10,hidden_dim=500):
        super(A_Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.a_encoder = nn.Sequential(
            nn.Linear(image_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 300),
            nn.BatchNorm1d(300),
            nn.LeakyReLU(0.2),
            nn.Linear(300,latent_dim*2))
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self,x):
        q = self.a_encoder(x)
        mu = q[:,:self.latent_dim]
        log_var = q[:,self.latent_dim:]
        z = self.reparameterize(mu,log_var)
        return mu,log_var,z

class A_Decoder(nn.Module):
    def __init__(self,image_size=784,latent_dim=10,hidden_dim=500):
        super(A_Decoder, self).__init__()
        self.a_decoder = nn.Sequential(
            nn.Linear(latent_dim, 300),
            nn.ReLU(True),
            nn.Linear(300, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, image_size),
            nn.Sigmoid()
        )
    def forward(self, z):
        return self.a_decoder(z)

class Autoencoder(nn.Module):
    def __init__(self,a_encoder,a_decoder):
        super(Autoencoder, self).__init__()

        self.a_encoder = a_encoder
        self.a_decoder = a_decoder

    def forward(self, input):
        _,_,z = self.a_encoder(input)
        x = self.a_decoder(z)
        return z,x

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def train(epoch,model,image_size,latent_dim,prior='Gauss'):
    model.train()
    train_loss = list(range(len(train_loader)))
    train_model_kernel = 0
    train_encoder_kernel = 0
    train_cross_kernel = 0
    recon_loss = list(range(len(train_loader)))
    sinkhorn_solver = SinkhornSolver(epsilon=0.01,iterations=20)

    start_time = time.time()
    for batch_idx, (real_data, _) in enumerate(train_loader):
        real_data = real_data.to(device)
        real_data = real_data.type(torch.float32)
        optimizer.zero_grad()
        if prior == 'Gauss':
            latent_priors = multivariate_normal.MultivariateNormal(loc=torch.zeros(latent_dim),
                                                                   covariance_matrix=torch.eye(latent_dim)).\
                sample(sample_shape=(real_data.size()[0],)).to(device)
        else:
            latent_priors = Variable(
                -2*torch.rand(real_data.size()[0], latent_dim) + 1,
                requires_grad=False
            ).to(device)

        mu, logvar, latent_encoded = model.a_encoder(real_data.view(-1, image_size))
        decoded_data = model.a_decoder(latent_priors)
        latent_decoded_data,_,_ = model.a_encoder(decoded_data)
        reconstructed_data = model.a_decoder(latent_encoded)
        observable_error = ((real_data.view(-1,image_size) - reconstructed_data).pow(2).mean(-1)).mean()

        C1 = compute_cost(decoded_data,real_data.view(-1,image_size))
        C2 = compute_cost(decoded_data,reconstructed_data)
        C3 = compute_cost(latent_priors - latent_decoded_data, latent_encoded - mu)
        C4 = compute_cost(torch.zeros_like(reconstructed_data),real_data.view(-1,image_size) - reconstructed_data)
        loss,_ = sinkhorn_solver(decoded_data,real_data.view(-1,image_size),C=C1+C2+C3+C4)
        loss.backward()
        train_loss[batch_idx] = loss.item()
        recon_loss[batch_idx] = observable_error.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(real_data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item()))
    end_time = time.time()
    end_time = end_time - start_time
    print('====> Epoch: {} Average loss: {:.4f}, Model Kernel: {:.6f},Encoder Kernel: {:.6f}, Cross Kernel: {:.6f}, Observable Error: {:.6f} Time: {:.6f}'.format(
        epoch, np.array(train_loss).mean(0),train_model_kernel/batch_idx, train_encoder_kernel/batch_idx,train_cross_kernel/batch_idx,np.array(recon_loss).mean(0),end_time))

    return np.array(train_loss).mean(0)
if __name__ == "__main__":
    latent_dim = 50
    image_size = 784
    num_hidden = 500
    load_previous = False

    encoder = A_Encoder(latent_dim=latent_dim,image_size=image_size,hidden_dim=num_hidden).to(device)
    decoder = A_Decoder(latent_dim=latent_dim,image_size=image_size,hidden_dim=num_hidden).to(device)
    model = Autoencoder(encoder, decoder)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-5, patience=20, verbose=True)

    if load_previous:
        previous_epoch = 100
        model.load_state_dict(torch.load('models/mnist_wvi_'+str(previous_epoch)+'_480_50.model'))
        encoder.load_state_dict(torch.load('models/mnist_wvi_encoder_'+str(previous_epoch)+'_480_50.model'))
        decoder.load_state_dict(torch.load('models/mnist_wvi_decoder_'+str(previous_epoch)+'_480_50.model'))
    else:
        previous_epoch = 0
    training_loss = []
    for epoch in range(1, args.epochs + 1):
        training_loss.append(train(epoch,model,image_size=image_size,latent_dim=latent_dim,prior='Gauss'))
        if epoch % 50 == 0:
            torch.save(model.state_dict(),'models/mnist_wvi_' + str(epoch+previous_epoch) + '_' + str(args.batch_size) + '_' + str(latent_dim) +  '.model')
            torch.save(encoder.state_dict(), 'models/mnist_wvi_encoder_' + str(epoch+previous_epoch) + '_' + str(args.batch_size)+ '_' + str(latent_dim)+ '.model')
            torch.save(decoder.state_dict(), 'models/mnist_wvi_decoder_'+ str(epoch+previous_epoch) + '_' + str(args.batch_size)+ '_' + str(latent_dim)+ '.model')
        scheduler.step(training_loss[-1])

    plt.plot(np.array(training_loss))
    plt.savefig('results/losses.PNG')
    plt.close()
