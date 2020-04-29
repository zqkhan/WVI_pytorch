import torch
import numpy as np
from torchvision.utils import make_grid


def compute_cost(x, y,ground_metric=lambda x: torch.pow(x, 2)):
    x_ = x.unsqueeze(-2)
    y_ = y.unsqueeze(-3)
    C = torch.sum(ground_metric(x_ - y_), dim=-1)
    return C


def sinkhorn(C,t,eps=0.1,device=torch.cuda):

    K = torch.exp(-C/eps)
    # K = K + 1e-8
    n,m = C.shape
    K_tilde = (torch.mm(torch.eye(n).to(device) / n, K)).to(device)

    r = (torch.ones((n,1))/n).to(device)
    c = (torch.ones((m,1))/m).to(device)
    u_0 = r
    u_t = u_0
    for tau in range(t):
        a = torch.mm(K_tilde,u_t)
        b = c / a
        u_t = 1.0 / torch.mm(K_tilde, b)
        # u_t = m / (K @ b)


    v = c / torch.mm(K_tilde,u_t)
    S = torch.sum(torch.mm(K * C, v) * u_t)
    return S


def calculate_C1(x_1,x_2):
    x_1 = x_1.unsqueeze(1)
    x_2 = x_2.unsqueeze(0)
    return (x_1 - x_2).pow(2).sum(2).pow(.5)

def calculate_C(x_1,z_1,x_2,z_2,hqx_1,hqx_2,gpz_1,gpz_2,w_mode='1100'):

    if w_mode == '1000':
        C = calculate_C1(x_1,x_2)

    elif w_mode == '1100':
        C1 = calculate_C1(x_1,x_2)
        C_pb = calculate_C1(gpz_1,gpz_2)
        C = C1 + C_pb

    elif w_mode == '1110':
        C1 = calculate_C1(x_1,x_2)
        C_pb = calculate_C1(gpz_1,gpz_2)
        C_la = calculate_C1(z_1 - hqx_1,z_2 - hqx_2)
        C = C1 + C_pb + C_la

    elif w_mode == '1111':
        C1 = calculate_C1(x_1,x_2)
        C_pb = calculate_C1(gpz_1,gpz_2)
        C_la = calculate_C1(z_1 - hqx_1,z_2 - hqx_2)
        C_oa = calculate_C1(torch.zeros_like(x_1),x_2 - gpz_2)
        C = C1 + C_pb + C_la + C_oa
    else:

        print ("unexpected mode")
    return C



def save_image(tensor, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    # im.save(filename)
    return im

def vector_interpolate(A,B,alpha_step=0.1):
    N_vectors = int(1/alpha_step)
    c = torch.empty(size=(N_vectors + 1,len(A)))
    count = 0
    for a in np.arange(0,1+alpha_step,alpha_step):
        c[count,:] = a*A + (1-a)*B
        count += 1

    return c