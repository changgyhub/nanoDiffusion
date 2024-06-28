import os
import time

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn

from data import get_mnist_dataloader, get_mnist_tensor_shape
from sampler import DDPM, DDIM
from model import UNet

batch_size = 512
n_train_epochs = 100
device = "mps" if torch.backends.mps.is_available() else "cpu"
output_dir = 'out'
sampler_dict = {'ddpm': DDPM, 'ddim': DDIM}
n_steps = 1000
model_name = 'model_unet_res.pth'
train_model = True

def train(sampler, net, device:str):
    n_steps = sampler.n_steps
    dataloader = get_mnist_dataloader(batch_size)
    net = net.to(device)
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    tic = time.time()
    for e in range(n_train_epochs):
        total_loss = 0
        for x, _ in dataloader:  # x_0 ~ q(x_0)
            B = x.shape[0]
            x = x.to(device)
            t = torch.randint(0, n_steps, (B, )).to(device)  # t ~ Uniform({1, ..., T})
            eps = torch.randn_like(x).to(device)
            x_t = sampler.sample_forward(x, t, eps)
            eps_theta = net(x_t, t.reshape(B, 1))
            loss = mse_loss(eps_theta, eps)  # MSE(eps - eps_theta)
            optimizer.zero_grad()
            loss.backward()  # gradient descent step
            optimizer.step()
            total_loss += loss.item() * B
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(net.state_dict(), f"{output_dir}/{model_name}")
        print(f'epoch {e}, avg loss {total_loss}, elapsed {(toc - tic):.2f}s')


def generate_images(sampler, net, output_path:str, device:str, n_sample_per_side:int = 10):
    net = net.to(device).eval()
    with torch.no_grad():
        shape = (int(n_sample_per_side**2), *get_mnist_tensor_shape())
        samples = sampler.sample_backward(net, shape).detach()  # generate samples 
        samples = ((samples + 1) / 2) * 255 
        samples = samples.clamp(0, 255)
        samples = einops.rearrange(samples, '(b1 b2) c h w -> (b1 h) (b2 w) c', b1=n_sample_per_side)  # arrange samples to a square image
        image = samples.cpu().numpy().astype(np.uint8)  # default image coding
        cv2.imwrite(output_path, image)  # save the image


if __name__ == '__main__':
    net = UNet(n_steps)
    if train_model:
        train_sampler_str = "ddpm"
        os.makedirs(f"{output_dir}/{train_sampler_str}", exist_ok=True)
        train(sampler_dict[train_sampler_str](n_steps, device=device), net, device=device)
    else:
        net.load_state_dict(torch.load(f"{output_dir}/{model_name}"))
    for (sampler_str, sampler) in sampler_dict.items():
        generate_images(sampler(n_steps, device=device), net, f"{output_dir}/{sampler_str}/generate.png", device)
