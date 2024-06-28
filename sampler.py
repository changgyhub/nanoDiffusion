import os
import cv2
import einops
import numpy as np
import torch

from data import get_mnist_dataloader

def sqrt(x):
    if isinstance(x, torch.Tensor):
        return torch.sqrt(x)
    return x ** 0.5

class DDPM():

    def __init__(self, n_steps: int, device: str, min_beta: float = 0.0001, max_beta: float = 0.02):
        self.n_steps = n_steps
        self.device = device
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.empty_like(self.alphas)
        product = 1
        for i, alpha in enumerate(self.alphas):
            product *= alpha
            self.alpha_bars[i] = product

    def sample_forward(self, x_0, t, noise=None):
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        eps = torch.randn_like(x_0) if noise is None else noise
        res = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * eps  # x_t = sqrt(alpha_bar) * x_0  + sqrt(1 - alpha_bar) * eps
        return res

    def sample_backward(self, net, in_shape):
        x = torch.randn(in_shape).to(self.device)
        net = net.to(self.device)
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_t(net, x, t)  # x_{t-1} = sample_backward_t(x_t)
        return x  # x_0

    def sample_backward_t(self, net, x_t, t):
        eps_t = net(x_t, torch.tensor([t] * x_t.shape[0], dtype=torch.long).to(x_t.device).unsqueeze(1))
        mu_t = (x_t - (1 - self.alphas[t]) / sqrt(1 - self.alpha_bars[t]) * eps_t) / sqrt(self.alphas[t])  # posterior mean
        if t == 0:
            noise_t = 0
        else:
            beta_t = self.betas[t]
            beta_tilde_t =  (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t]) * beta_t  # posterior variance
            noise_t = sqrt(beta_tilde_t) * torch.randn_like(x_t)
        x_t_minus_1 = mu_t + noise_t  # x_{t-1} = N(x_t-1; mu(x_t, x_0), beta(_tilde)_t I)
        return x_t_minus_1
    
    
class DDIM(DDPM):

    def __init__(self, n_steps: int, device: str, min_beta: float = 0.0001, max_beta: float = 0.02):
        super().__init__(n_steps, device, min_beta, max_beta)

    def sample_backward(self, net, in_shape, ddim_step_ratio=0.1, eta=0):
        n_ddim_steps = int(ddim_step_ratio * self.n_steps)
        s = torch.linspace(self.n_steps, 0, (n_ddim_steps + 1)).to(self.device).to(torch.long)
        x = torch.randn(in_shape).to(self.device)
        net = net.to(self.device)
        for i in range(n_ddim_steps):
            x = self.sample_backward_t(net, x, s, i+1, eta)  # x_{s-1} = sample_backward_t(x_s)
        return x
    
    def sample_backward_t(self, net, x_t, s, i, eta):
        t = s[i-1] - 1
        t_minus_1 = s[i] - 1
        alpha_bar_t = self.alpha_bars[t]
        alpha_bar_t_minus_1 = self.alpha_bars[t_minus_1] if t_minus_1 >= 0 else 1

        eps_t = net(x_t, torch.tensor([t] * x_t.shape[0], dtype=torch.long).to(x_t.device).unsqueeze(1))
        var_t = (eta ** 2) * (1 - alpha_bar_t_minus_1) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_minus_1)
    
        predicted_x_0 = sqrt(alpha_bar_t_minus_1) * (x_t - sqrt(1 - alpha_bar_t) * eps_t) / sqrt(alpha_bar_t)
        direction_pointing_to_x_t = sqrt(1 - alpha_bar_t_minus_1 - var_t) * eps_t
        noise_t = sqrt(var_t) * torch.randn_like(x_t)
        x_t_minus_1 = predicted_x_0 + direction_pointing_to_x_t + noise_t
        return x_t_minus_1
        

def visualize_adding_noise(n_steps, output_dir, sampler_dict, device):
    dataloader = get_mnist_dataloader(5)
    noise_percents = torch.linspace(0, 0.999, 30)
    x, _ = next(iter(dataloader))
    x = x.to(device)

    for (sampler_str, sampler) in sampler_dict.items():
        sampler_obj = sampler(n_steps, device=device)
        x_ts = []
        for noise_percent in noise_percents:
            t = torch.tensor([int(n_steps * noise_percent)]).unsqueeze(1)
            x_t = sampler_obj.sample_forward(x, t)
            x_ts.append(x_t)
        x_ts = torch.stack(x_ts, 0)
        x_ts = ((x_ts + 1) / 2) * 255
        x_ts = x_ts.clamp(0, 255)
        x_ts = einops.rearrange(x_ts, 'n1 n2 c h w -> (n2 h) (n1 w) c')
        image = x_ts.cpu().numpy().astype(np.uint8)

        os.makedirs(f"{output_dir}/{sampler_str}", exist_ok=True)
        cv2.imwrite(f"{output_dir}/{sampler_str}/adding_noise.png", image)


if __name__ == '__main__':
    n_steps = 1000
    output_dir = 'out'
    sampler_dict = {'ddpm': DDPM, 'ddim': DDIM}
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    visualize_adding_noise(n_steps, output_dir, sampler_dict, device)
