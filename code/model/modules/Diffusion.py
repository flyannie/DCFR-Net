import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from functools import partial
from tqdm.contrib import tzip
from utils.utils import default, compute_alpha


class Diffusion(nn.Module):
    def __init__(self, encoder, denoise_fn, image_size, channels=3, conditional=True, feat_unfold=False, local_ensemble=False, cell_decode=False, schedule_opt=None):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.encoder = encoder
        self.denoise_fn = denoise_fn
        self.conditional = conditional 
        self.feat_unfold = feat_unfold
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode

    def set_loss(self, device):
        self.loss_func = nn.L1Loss(reduction='sum').to(device)

    def set_new_noise_schedule(self, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = np.linspace(1e-6, 1e-2, 2000, dtype=np.float64)
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas 
        alphas_cumprod = np.cumprod(alphas, axis=0) 
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod)) 
        timesteps, = betas.shape 
        self.num_timesteps = int(timesteps) 
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        self.register_buffer('ddim_c1', torch.sqrt(to_torch((1. - alphas_cumprod / alphas_cumprod_prev) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod))))

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)

    def p_losses(self, x_in, noise=None):
        inp, scaler = x_in['lr'], x_in['scaler']
        x_feat = self.gen_feat(inp, x_in['hr'].shape[2:])
        x_con = x_feat
        x_start = x_in['hr']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1) 
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(np.random.uniform(self.sqrt_alphas_cumprod_prev[t-1], self.sqrt_alphas_cumprod_prev[t], size=b)).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)
        noise = default(noise, lambda: torch.randn_like(x_start)) 
        x_noisy = self.q_sample(x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)
        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(torch.cat([x_con, x_noisy], dim=1), x_con, scaler, continuous_sqrt_alpha_cumprod)
        loss = self.loss_func(noise, x_recon)
        return loss

    def gen_feat(self, inp, shape):
        feat = self.encoder(inp, shape) 
        return feat

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (continuous_sqrt_alpha_cumprod * x_start + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False, use_ddim=False):
        if not use_ddim:
            return self.p_sample_loop(x_in, continous)
        else:
            return self.generalized_steps(x_in, conditional_input=None, continous=continous)

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps // 10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x, scaler = x_in['lr'], x_in['scaler']
            shape = x.shape
            gt_shape = list(x_in['hr'].shape)
            img = torch.randn(gt_shape, device=device)
            x_feat = self.gen_feat(x, gt_shape[2:])
            x_con = x_feat
            ret_img = x_con
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, scaler, condition_x=x_con)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def p_sample(self, x, t, scaler, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(x=x, t=t, scaler=scaler, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    def p_mean_variance(self, x, t, scaler, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), condition_x, scaler, noise_level))
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, noise_level))
        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    @torch.no_grad()
    def generalized_steps(self, x_in, conditional_input=None, continous=False):
        device = self.betas.device
        skip = self.num_timesteps // 200  
        seq = range(0, self.num_timesteps, skip)
        seq_next = [-1] + list(seq[:-1])
        x, scaler = x_in['lr'], x_in['scaler']
        b = x.size(0)
        gt_shape = list(x_in['hr'].shape)
        img = torch.randn(gt_shape, device=device)
        x_feat = self.gen_feat(x, gt_shape[2:]) 
        conditional_input = x_feat
        ret_img = img
        for i, j in tzip(reversed(seq), reversed(seq_next)):
            if i == 0:
                break
            noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[i + 1]]).repeat(b, 1).to(x.device) 
            t = (torch.ones(b) * i).to(x.device)
            next_t = (torch.ones(b) * j).to(x.device)
            at = compute_alpha(self.betas, t.long())
            at_next = compute_alpha(self.betas, next_t.long())
            xt = ret_img[-1] # [c,h,w]
            et = self.denoise_fn(torch.cat([conditional_input, xt.unsqueeze(0)], dim=1), conditional_input, scaler, noise_level)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            c1 = (0 * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt())
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(img) + c2 * et
            ret_img = torch.cat([ret_img, xt_next], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]


