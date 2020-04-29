from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.autograd import Variable
from utils.helpers import save_image,vector_interpolate
from torch.distributions import normal,multivariate_normal
import pickle
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import numpy as np
# from main_bigan_mnsit_model import Autoencoder,A_Encoder,A_Decoder
from main_mnist import Autoencoder,A_Encoder,A_Decoder
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from PIL import Image

save_images = True
do_tsne = True
gauss_prior = False


NUM_COLORS = 25
cm = plt.get_cmap('gist_rainbow')
color_list = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS) ]

# color_file = 'color_chart.txt'
# R = np.loadtxt(color_file, delimiter='\t', usecols=0)/255
# G = np.loadtxt(color_file, delimiter='\t', usecols=1)/255
# B = np.loadtxt(color_file, delimiter='\t', usecols=2)/255
# color_list = [(r, g, b) for r, g, b in zip(R, G, B)]

# plt.scatter(data[:,0],data[:,1],c=labels)
# plt.show()

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=True,
help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
dataset_dimension = 28

KERNEL = 'gauss'

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


test_loader = torch.utils.data.DataLoader(
datasets.MNIST('../data', train=False, download=True,
transform=transforms.Compose([transforms.ToTensor()])),
batch_size=args.batch_size, shuffle=True, ** kwargs)

training_dataset = datasets.MNIST(root='../data',train=True,transform=transforms.Compose([
    transforms.Resize((dataset_dimension,dataset_dimension)),transforms.ToTensor(),
                   ]))

test_dataset = datasets.MNIST(root='../data',train=False,transform=transforms.Compose([
    transforms.Resize((dataset_dimension,dataset_dimension)),transforms.ToTensor(),
                   ]))


latent_dim = 50
image_size = 784
num_hidden = 500
epoch = 100
batch_size = 512
gauss_prior = True

encoder = A_Encoder(latent_dim=latent_dim, image_size=image_size, hidden_dim=num_hidden).to(device)
decoder = A_Decoder(latent_dim=latent_dim, image_size=image_size, hidden_dim=num_hidden).to(device)
model = Autoencoder(encoder, decoder)
clf = TSNE(n_components=2,perplexity=100)
if __name__ == "__main__":

    model.load_state_dict(torch.load('models/mnist_wvi_' + str(epoch) + '_' + str(batch_size) + '_' + str(latent_dim) + '.model'))
    encoder.load_state_dict(torch.load('models/mnist_wvi_encoder_' + str(epoch) + '_' + str(batch_size) + '_' + str(latent_dim) + '.model'))
    decoder.load_state_dict(torch.load('models/mnist_wvi_decoder_' + str(epoch) + '_' + str(batch_size) + '_' + str(latent_dim) + '.model'))
    model.eval()
    testing_data = test_dataset.data.view(-1,image_size).to(device)
    testing_data = testing_data.type(torch.float32)
    testing_data = testing_data/255
    latent_data,_,_ = encoder(testing_data)

    reconstructed_data = decoder(latent_data)

    observable_error = ((testing_data - reconstructed_data).pow(2).mean(-1)).mean()
    observable_error_std = ((testing_data - reconstructed_data).pow(2).mean(-1)).std()

    latent_priors = multivariate_normal.MultivariateNormal(loc=torch.zeros(latent_dim),
                                                           covariance_matrix=torch.eye(latent_dim)). \
        sample(sample_shape=(10000,)).to(device)
    generate_data = decoder(latent_priors)
    latent_data_reconstructed,_,_ = encoder(generate_data)

    loss = nn.MSELoss()
    latent_loss = []
    for i in range(len(latent_priors)):
        latent_loss.append(loss(latent_priors[i, :], latent_data_reconstructed[i, :]).data.numpy())

    latent_error = np.array(latent_loss).mean()
    latent_error_std = np.array(latent_loss).std()
    print ('pixelwise mean squared error:' + str(observable_error.data) + u"\u00B1" + str(observable_error_std.data))
    print ('latent_error:' + str(latent_error) + u"\u00B1" + str(latent_error_std))

    training_data = training_dataset.data.view(-1, image_size).to(device)
    training_data = training_data.type(torch.float32)
    training_data = training_data / 255
    training_data, _, _ = encoder(training_data)


    clf_nn = KNeighborsClassifier(n_neighbors=1, weights='uniform')
    clf_nn.fit(training_data.detach().numpy(), training_dataset.train_labels.numpy())
    predict_labels = clf_nn.predict(latent_data.detach().numpy())
    test_labels = test_dataset.test_labels.numpy()
    print(np.sum(predict_labels == test_labels) / len(predict_labels))
    count = 0
    zero_matrix = torch.zeros([128,image_size])
    clf_test = NearestNeighbors(n_neighbors=1,n_jobs=-1)
    clf_test.fit(testing_data.data.numpy())
    for (real_data, labels) in (test_loader):
        if gauss_prior:
            latent_priors = multivariate_normal.MultivariateNormal(loc=torch.zeros(latent_dim),
                                                                   covariance_matrix=torch.eye(latent_dim)).\
                sample(sample_shape=(64,)).to(device)
        else:
            latent_priors = Variable(
                -2 * torch.rand(64, latent_dim) + 1,
                requires_grad=False
            ).to(device)
        decoded_data = decoder(latent_priors)
        distances = clf_test.kneighbors(decoded_data.data.numpy())[0]
        if save_images:
            im = save_image(decoded_data.view(64, 1, dataset_dimension, dataset_dimension))
            im.save('results/sample_test_' + str(count) + '.png')
        print ('Iteration: ' + str(count))
        real_data = real_data.to(device)
        real_data = real_data.view(-1,image_size)
        mu, logvar, latent_encoded = encoder(real_data)
        latent_encoded_tsne = clf.fit_transform(latent_encoded.detach().numpy())
        string_labels = [str(l.numpy()) for l in labels]
        if do_tsne:
            for p in range(5,6,5):
                clf = TSNE(n_components=2,perplexity=25)
                latent_mu_tsne = clf.fit_transform(mu.detach().numpy())
                embed = latent_mu_tsne
                plt.figure()
                ax = plt.subplot(111)
                ax_min = np.min(embed, 0)
                ax_max = np.max(embed, 0)

                ax_dist_sq = np.sum((ax_max - ax_min) ** 2)
                ax.axis('off')
                shown_images = np.array([[1., 1.]])
                for i in range(embed.shape[0]):
                    dist = np.sum((embed[i] - shown_images) ** 2, 1)
                    if np.min(dist) < 3e-4 * ax_dist_sq:  # don't show points that are too close
                        continue
                    shown_images = np.r_[shown_images, [embed[i]]]
                    imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(np.reshape(real_data.view(args.batch_size, 1, dataset_dimension, dataset_dimension)[i, ...], [28, 28]),
                                                                              zoom=0.6, cmap=plt.cm.gray_r),
                                                        xy=embed[i], frameon=False)
                    ax.add_artist(imagebox)

                plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
                plt.xticks([]), plt.yticks([])

                if save_images:
                    plt.savefig('results/latent_space_tsne_' + str(count) + '.png')
                else:
                    plt.show()
                plt.close()
        else:
            embed = mu.detach().numpy()
            plt.figure()
            ax = plt.subplot(111)
            ax_min = np.min(embed, 0)
            ax_max = np.max(embed, 0)

            ax_dist_sq = np.sum((ax_max - ax_min) ** 2)
            ax.axis('off')
            shown_images = np.array([[1., 1.]])
            for i in range(embed.shape[0]):
                dist = np.sum((embed[i] - shown_images) ** 2, 1)
                if np.min(dist) < 3e-4 * ax_dist_sq:  # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [embed[i]]]
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(
                    np.reshape(real_data.view(args.batch_size, 1, dataset_dimension, dataset_dimension)[i, ...],
                               [28, 28]),
                    zoom=0.6, cmap=plt.cm.gray_r),
                                                    xy=embed[i], frameon=False)
                ax.add_artist(imagebox)

            plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
            plt.xticks([]), plt.yticks([])

            if save_images:
                plt.savefig('results/latent_space_' + str(count) + '.png')
            else:
                plt.show()
            plt.close()
        if count == 0:
            latent1 = mu[55, :]
            latent2 = mu[22, :]
            inter_vectors = vector_interpolate(latent1, latent2, alpha_step=1 / 23)
            inter_vectors = decoder(inter_vectors)
            if save_images:
                im0 = save_image(inter_vectors.view(len(inter_vectors), 1, dataset_dimension, dataset_dimension),nrow=24)
                combined_im = Image.new('RGB', (im0.width, im0.height))
                combined_im.paste(im0)
                combined_im.save('results/mnist_interpolations.png')
        reconstructions = decoder(mu)
        if save_images:
            plot_real_data = real_data[:24,:]
            plot_reconstruction_data = reconstructions[:24,:]
            if save_images:
                im0 = save_image(plot_real_data.view(len(plot_real_data), 1, dataset_dimension, dataset_dimension),nrow=24)
                combined_im = Image.new('RGB', (im0.width, im0.height))
                combined_im.paste(im0)
                combined_im.save('results/mnist_real_row_' + str(count) + '.png')
                im0 = save_image(plot_reconstruction_data.view(len(plot_reconstruction_data),
                                                               1, dataset_dimension, dataset_dimension),nrow=24)
                combined_im = Image.new('RGB', (im0.width, im0.height))
                combined_im.paste(im0)
                combined_im.save('results/mnist_reconstruction_row_' + str(count) + '.png')
        if save_images:
            for i in range(0, 64, 16):
                if i == 0:
                    zero_matrix[i:i + 8, :] = real_data[i:i + 8, :]
                    zero_matrix[i + 8:i + 16, :] = reconstructions[i:i + 8, :]
                else:
                    zero_matrix[i:i + 8, :] = real_data[i-8:i,:]
                    zero_matrix[i + 8:i + 16, :] = reconstructions[i-8:i,:]
            real_data = zero_matrix[:64,:]
            reconstructions = zero_matrix[64:,:]
            im1 = save_image(real_data.view(64, 1, dataset_dimension, dataset_dimension))
            im2 = save_image(reconstructions.view(64, 1, dataset_dimension, dataset_dimension))
            combined_im = Image.new('RGB',(im1.width,im1.height))
            x_offset = 0
            for im in [im1,im2]:
                combined_im.paste(im,(x_offset,0))
                x_offset += im.size[0]
            combined_im.save('results/real_recon_' + str(count) + '.png')
        #
        count +=1
        if count == 10:
            break
    r = 3




