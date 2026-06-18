"""Conditional DDPM and antithetic sampler for the **diffusion** pipeline.

Faithful migration of the original `ddpm_conditional.py`. The module is
reproduced verbatim -- the noise-schedule construction, the forward
(`noise_signal`) process, and the reverse sampling loops (`inference` and
`inference_antithetic`) are byte-for-byte unchanged so that published results
remain reproducible. No imports needed rewriting (the file depends only on
third-party packages), and there were no hard-coded filesystem paths to route
through :mod:`cardiac_map_diffusion.paths`; only this module docstring was
expanded.

    - This file contains the implementation of the conditional diffusion model.

    - This class contains the implementation of the antithetic sampling method.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    ''' This is the main class for the conditional diffusion model.
        It provides most of the functions needed for training and inference.'''

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, signal_len=370, noise_schedule="linear", device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.signal_len = signal_len
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.beta = self.prepare_noise_schedule(noise_schedule).to(self.device)  # dim: [noise_steps]
        self.alpha = 1 - self.beta                                               # dim: [noise_steps]
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)                        # dim: [noise_steps]

    def prepare_noise_schedule(self, noise_schedule):
        if noise_schedule == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        if noise_schedule == "quadratic":
            return torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.noise_steps) ** 2
    
    def noise_signal(self, x, t):
        ''' Function to add noise to a signal, given the noise level t.
            Input: 
                x: clean signal x_0, dim[batch_size, channels (=1), signal_len]
                t: noise level dim[batch_size]
            Output:
                x_noisy: noisy signal dim[batch_size, channels, signal_len]
                noise: noise dim[batch_size, channels, signal_len]'''
           
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]                   # dim: [batch_size, 1, 1]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]     # dim: [batch_size, 1, 1]
        epsilon = torch.randn_like(x)                                                   # dim: [batch_size, channels, signal_len]

        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon         # dim: [batch_size, channels, signal_len]
    
    def sample_timesteps(self, n):
        ''' Function to randomly sample noise levels for a batch of signals.
        '''
        return torch.randint(low=1, high=self.noise_steps, size=(n,)) # dim: [batch_size]
    
    def inference(self, model, n, x_noisy):
        ''' Function to sample a batch of denoised signals.
            Input: 
                model: denoising model
                n: number of samples, NOT IMPLEMENTED SET n=1
                x_noisy: noisy signal dim[batch_size, channels, signal_len]
            Output:
                x: denoised signal dim[batch_size, channels, signal_len]
                '''
        batch_size = x_noisy.size(0)
        logging.info(f"Denoise {batch_size} signals...")
        model.eval()
        with torch.no_grad():
            x_t = torch.randn((batch_size, 1, self.signal_len)).to(self.device)            # dim: [batch,1,signal_len] random gaussian noise as starting point
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                sqrt_alpha_hat = torch.sqrt(self.alpha_hat[i])                             # dim: [scalar]
                sqrt_alpha_hat = (torch.ones(batch_size).to(self.device) * sqrt_alpha_hat
                                  )                                        # dim: [batch] 
                
                predicted_noise = model(x_t, x_noisy, sqrt_alpha_hat)                      # dim: [batch, channels, signal_len]

                alpha = self.alpha[i]                                                    # dim: [scalar]
                alpha_hat = self.alpha_hat[i]                                            # dim: [scalar]
                beta = self.beta[i]                                                      # dim: [scalar]                               

                # Do not add noise in the last time step t_1 -> t_0
                if i > 1: 
                    noise = torch.randn_like(x_t)                                        # dim: [batch, channels, signal_len]
                else: 
                    noise = torch.zeros_like(x_t)
                
                sigma_t = torch.sqrt(beta * (1. - self.alpha_hat[i-1]) / (1. - alpha_hat)) # dim: [scalar]

                x_t = 1 / torch.sqrt(alpha) * (x_t - ((1-alpha) / (torch.sqrt(1-alpha_hat)) * predicted_noise)) + sigma_t * noise 
 
        model.train()

        return x_t
    
    def inference_antithetic(self, model, n, x_noisy):
        ''' Function to sample a batch of denoised signals and its 'antithetic' counterpart.
            Input: 
                model: denoising model
                n: number of samples (eventually this was set to 1 as the N-shots is done by calling this function multiple times)
                x_noisy: noisy signal dim[batch_size, channels, signal_len]
            Output:
                x: denoised signal dim[batch_size, channels, signal_len]
                x_anti: antithetic counterpart of the denoised signal dim[batch_size, channels, signal_len]
                '''
        batch_size = x_noisy.size(0)
        logging.info(f"Denoise {batch_size} signals...")
        model.eval()

        with torch.no_grad():
            # random gaussian noise as starting point
            x_t = torch.randn((batch_size, 1, self.signal_len)).to(self.device)          # dim: [batch,1,signal_len] 
            x_t_anti = torch.neg(x_t)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                sqrt_alpha_hat = torch.sqrt(self.alpha_hat[i])                           # dim: [scalar]
                sqrt_alpha_hat = (torch.ones(batch_size).to(self.device) * sqrt_alpha_hat) # dim: [batch] 
                
                predicted_noise = model(x_t, x_noisy, sqrt_alpha_hat)                    # dim: [batch, channels, signal_len]
                predicted_noise_anti = model(x_t_anti, x_noisy, sqrt_alpha_hat)          # dim: [batch, channels, signal_len]

                alpha = self.alpha[i]                                                    # dim: [scalar]
                alpha_hat = self.alpha_hat[i]                                            # dim: [scalar]
                beta = self.beta[i]                                                      # dim: [scalar]                               

                # Do not add noise in the last time step t_1 -> t_0
                if i > 1: 
                    noise = torch.randn_like(x_t)                                        # dim: [batch, channels, signal_len]
                    noise_anti = torch.neg(noise)                                        # dim: [batch, channels, signal_len]
                else: 
                    noise = torch.zeros_like(x_t)
                    noise_anti = torch.zeros_like(x_t_anti)   
                
                sigma_t = torch.sqrt(beta * (1. - self.alpha_hat[i-1]) / (1. - alpha_hat)) # dim: [scalar]

                x_t = 1 / torch.sqrt(alpha) * (x_t - ((1-alpha) / (torch.sqrt(1-alpha_hat)) * predicted_noise)) + sigma_t * noise
                x_t_anti = 1 / torch.sqrt(alpha) * (x_t_anti - ((1-alpha) / (torch.sqrt(1-alpha_hat)) * predicted_noise_anti)) + sigma_t * noise_anti 
 
        model.train()

        return x_t, x_t_anti

    
''' This class is used for keeping track of the moving average of the weights. (Was not used in the final experiments.)
'''
class ExponentialMovingAverage:
    def __init__(self, decay_rate):
        super().__init__()
        self.decay_rate = decay_rate
        self.iteration = 0

    def update_parameters(self, target_model, source_model):
        for source_param, target_param in zip(source_model.parameters(), target_model.parameters()):
            target_param.data = self.compute_average(target_param.data, source_param.data)

    def compute_average(self, previous_value, new_value):
        if previous_value is None:
            return new_value
        return previous_value * self.decay_rate + (1 - self.decay_rate) * new_value

    def apply_ema(self, ema_model, model, start_step=2000):
        if self.iteration < start_step:
            self.initialize_parameters(ema_model, model)
            self.iteration += 1
            return
        self.update_parameters(ema_model, model)
        self.iteration += 1

    def initialize_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())