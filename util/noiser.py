import torch, math, numpy as np


def check_size(x, coef):
    if isinstance(coef, (int, float)):
        return coef
    elif isinstance(coef, dict):
        for k, v in coef.items():
            if isinstance(v, torch.Tensor):
                while len(v.shape) < len(x.shape):
                    v = v.unsqueeze(-1)
                coef[k] = v
    elif isinstance(coef, torch.Tensor):
        while len(coef.shape) < len(x.shape):
            coef = coef.unsqueeze(-1)
    return coef


class BaseNoiser(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device

        self.training_timesteps = self.args.training_timesteps
        self.training_timestep_map = torch.arange(start=0, end=self.training_timesteps, step=1, dtype=torch.int64, device=self.device)
        self.inference_timesteps = self.args.inference_timesteps
        self.inference_timestep_map = torch.arange(start=0, end=self.training_timesteps, step=int(self.training_timesteps / self.inference_timesteps), dtype=torch.int64, device=self.device)

        self.num_timesteps = self.training_timesteps
        self.timestep_map = self.training_timestep_map

    def train(self, mode=True):
        self.num_timesteps = self.training_timesteps if mode else self.inference_timesteps
        self.timestep_map = self.training_timestep_map if mode else self.inference_timestep_map
    
    def eval(self):
        self.train(mode=False)
    
    def coefficient(self, t):
        raise NotImplementedError

    def forward(self, x_0, x_1, t, ode=True):
        
        coef = check_size(x_0, self.coefficient(t))
        x_t = coef['coef0'] * x_0 + coef['coef1'] * x_1
        if 'var' in coef and not ode:
            x_t = x_t + torch.randn_like(x_t) * coef['var'] ** 0.5
        return x_t

    def forward_dsb(self, x, x_0, x_1, t):
        #print('self.coefficient(t):', self.coefficient(t))
        coef_t = check_size(x, self.coefficient(t))
        #print('coef_t:', coef_t)
        #coef_t_plus_one = check_size(x, self.coefficient(t + 1))
        #x = x_0 + (x - x_0) / coef_t_plus_one['coef1'] * coef_t['coef1']
        x = coef_t['mu_x0'] * x_0 + coef_t['mu_y']* x_1 +coef_t['std']*torch.randn_like(x)
        return x
    
    def prepare_gamma_dsb(self):
        if hasattr(self, 'gammas'):
            return
        if self.args.gamma_type == "linear":
            gamma_max = 0.1
            gamma_min = 0.0001
            # linearly gamma_min -> gamma_max -> gamma_min
            self.gammas = torch.linspace(gamma_min, gamma_max, self.num_timesteps // 2, device=self.device)
            self.gammas = torch.cat([self.gammas, self.gammas.flip(dims=(0,))], dim=0)
        elif self.args.gamma_type.startswith("linear_"):
            gamma_min = float(self.args.gamma_type.split("_")[1])
            gamma_max = float(self.args.gamma_type.split("_")[2])
            # linearly gamma_min -> gamma_max -> gamma_min
            self.gammas = torch.linspace(gamma_min, gamma_max, self.num_timesteps // 2, device=self.device)
            self.gammas = torch.cat([self.gammas, self.gammas.flip(dims=(0,))], dim=0)
        elif self.args.gamma_type == "constant":
            self.gammas = 0.0005 * torch.ones(size=(self.num_timesteps,), dtype=torch.float32, device=self.device)
        elif self.args.gamma_type.startswith("constant_"):
            gamma = float(self.args.gamma_type.split("_")[1])
            self.gammas = gamma * torch.ones(size=(self.num_timesteps,), dtype=torch.float32, device=self.device)
        elif self.args.gamma_type == "increase":
            gamma_max = 0.5
            gamma_min = 0.0001
            # linearly gamma_min -> gamma_max -> gamma_min
            self.gammas = torch.linspace(gamma_min, gamma_max, self.num_timesteps , device=self.device)
        elif self.args.gamma_type == "increase_":
            gamma_min = float(self.args.gamma_type.split("_")[1])
            gamma_max = float(self.args.gamma_type.split("_")[2])
            # linearly gamma_min -> gamma_max -> gamma_min
            self.gammas = torch.linspace(gamma_min, gamma_max, self.num_timesteps , device=self.device)
        else:
            raise NotImplementedError(f"gamma_type {self.args.gamma_type} not implemented")


class LinearNoiser_selfrdb(BaseNoiser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        beta_start, beta_end = 0.1, 0.3
        gamma = 1
        betas_len = self.num_timesteps+1
        #print(beta_start, beta_end, betas_len)
        betas =  np.linspace(beta_start**0.5, beta_end**0.5, betas_len)**2
        betas = np.append(0., betas).astype(np.float32)
        if betas_len % 2 == 1:
            betas = np.concatenate([
                betas[:betas_len//2],
                [betas[betas_len//2]],
                np.flip(betas[:betas_len//2])
            ])
        
        else:
            betas = np.concatenate([
                betas[:betas_len//2],
                np.flip(betas[:betas_len//2])
            ])
        #print('betas:', betas)
        s = np.cumsum(betas)**0.5
        s_bar = np.flip(np.cumsum(betas))**0.5
        s = torch.tensor(s).to(self.device)
        s_bar = torch.tensor(s_bar).to(self.device)
        self.mu_x0, self.mu_y, _ = self.gaussian_product(s, s_bar)
        gamma = gamma * betas.sum()
        self.std = gamma * s / (s**2 + s_bar**2)   
        self.s = s
        #print('self.s:', s, self.mu_x0, self.mu_y)

    def gaussian_product(self, sigma1, sigma2):
        denom = sigma1**2 + sigma2**2
        mu1 = sigma2**2 / denom
        mu2 = sigma1**2 / denom
        var = (sigma1**2 * sigma2**2) / denom
        return mu1, mu2, var

    def coefficient(self, t):
        tmax = t.max() if isinstance(t, torch.Tensor) else t
        if tmax >= len(self.timestep_map):   
            ret = {
                'mu_x0': 0,
                'mu_y': 1,
                'std':0,
            }
        else:
            t = self.timestep_map.flip(dims=(0,))[t]
            ret = {
                'mu_x0': self.mu_x0[t],
                'mu_y': self.mu_y[t],
                'std': self.std[t],
            }
        ret = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in ret.items()}
        #print(ret)
        return ret

    def q_sample(self, t, x0, y):
        """ Sample q(x_t | x_0, y) """
        shape = [-1] + [1] * (x0.ndim - 1)

        mu_x0 = self.mu_x0[t].view(shape)
        mu_y = self.mu_y[t].view(shape)
        std = self.std[t].view(shape)

        x_t = mu_x0*x0 + mu_y*y + std*torch.randn_like(x0)
        
        return x_t.detach()

    def trajectory(self, x_0, x_1, sample=False):
        ones = torch.ones(size=(x_0.shape[0],), dtype=torch.int64, device=self.device)
        x_cache, gt_cache, t_cache, x0 = [], [], [], []
        x = x_0
        with torch.no_grad():
            for t in range(self.args.training_timesteps):
                t = t+1
                x_old = x.clone()
                t_old = self.q_sample(t, x_old, x_1)
                
                x_cache.append(t_old.clone())
                gt_cache.append(x_1)
            
                t_cache.append(ones * t)
                x0.append(x_0)
        x_cache = torch.stack( x_cache+[x_0], dim=0).cpu() if sample else torch.cat(x_cache, dim=0).cpu()
        gt_cache = torch.cat(gt_cache, dim=0).cpu()
        t_cache = torch.cat(t_cache, dim=0).cpu()
        x0 = torch.cat(x0, dim=0).cpu()
        return x_cache, x0, t_cache, gt_cache

class LinearNoiser(BaseNoiser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        beta_start, beta_end = 0.0001, 0.02

        betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float64, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def coefficient(self, t):
        tmax = t.max() if isinstance(t, torch.Tensor) else t
        if tmax >= len(self.timestep_map):   
            ret = {
                'coef0': 0,
                'coef1': 1,
            }
        else:
            t = self.timestep_map.flip(dims=(0,))[t]
            ret = {
                'coef0': self.sqrt_one_minus_alphas_cumprod[t],
                'coef1': self.sqrt_alphas_cumprod[t],
            }
        ret = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in ret.items()}
        #print('ret:', ret)
        return ret

def create_noiser(name, *args, **kwargs):
    name = name.lower()
    if 'flow' in name:
        noiser = FlowNoiser
    elif 'linear' in name:
        noiser = LinearNoiser
    elif 'cosing' in name:
        noiser =  CosingNoiser
    elif 'dsb' in name:
        noiser = DSBNoiser
    elif 'self' in name:
        print('use selfrdb noiser')
        noiser = LinearNoiser_selfrdb
    else:
        raise NotImplementedError
    
    return noiser(*args, **kwargs)
