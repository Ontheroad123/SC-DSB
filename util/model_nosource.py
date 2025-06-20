import torch
from functools import partial
from util.network import ResMLP, DhariwalUNet
from util.vision_transformer import VisionTransformer
from backbones.ncsnpp import NCSNpp

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


class BaseModel(torch.nn.Module):
    def __init__(self, args, device, noiser, rank):
        super().__init__()
        self.args = args
        self.device = device
        self.noiser = noiser
        self.rank = rank

        if self.args.network == 'mlp':
            self.network = ResMLP(dim_in=2, dim_out=2, dim_hidden=128, num_layers=5, n_cond=self.noiser.training_timesteps)
        elif self.args.network == 'selfrdb':
            print('create ncsnpp!')
            self.network = NCSNpp()
        else:
            
            img_resolution = 32
            if "-512" in args.dataset:
                img_resolution = 64
                
            in_channels = 4
            if self.args.dataset == "afhq-cat-64" or self.args.dataset == "celeba-64":
                in_channels = 3

            if args.network == 'adm':
                self.network = DhariwalUNet(
                    img_resolution=img_resolution,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    label_dim=0,
                    augment_dim=0,
                    model_channels=96,
                    channel_mult=[1,2,2,2],
                    channel_mult_emb=4,
                    num_blocks=4,
                    attn_resolutions=[8, 4],
                    dropout=0.10,
                    label_dropout=0,
                )
            elif args.network == 'uvit-b':
                self.network = VisionTransformer(
                    img_size=64,
                    patch_size=4, embed_dim=512, 
                    num_heads=8, mlp_ratio=4, qkv_bias=True,
                    depth=13,
                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), 
                    num_classes=-1,
                    use_fp16=self.args.use_amp,
                )

        self.network.to(self.device)
        self.network = torch.nn.parallel.DistributedDataParallel(self.network, device_ids=[self.device], output_device=self.device) if (hasattr(self.args, 'gpus') and self.args.gpus > 1) else self.network

    def target(self, x_0, x_1, x_t, t):
        raise NotImplementedError

    def forward(self, x_t, t, x0_r):
        #t = self.noiser.timestep_map[t]
        #print('network input:', x_t.shape, x_t.min(), x_t.max(), t,self.noiser.timestep_map)
        x = self.network(x_t, t, x0_r)
        #print('network output:', x.shape, x.min(), x.max())
        return x

    def predict_boundary(self, x_0, x_t, t):
        raise NotImplementedError
    
    def predict_next(self, x_0, x_t, t):
        x_0, x_1 = self.predict_boundary(x_0, x_t, t)
        x_t = self.noiser(x_0, x_1, t + 1)
        return x_t

    def inference(self, x_0, return_all=False):
        self.eval()
        x_t_all = [x_0.clone()]
        with torch.no_grad():
            x_t = x_0
           
            ones = torch.ones(size=(x_t.shape[0],), dtype=torch.int64, device=x_t.device)
            for t in range(self.noiser.num_timesteps):
                with torch.autocast(device_type="cuda", enabled=self.args.use_amp):
                    x_t = self.predict_next(x_0, x_t, ones * t)
                x_t = x_t.float()
                if return_all:
                    x_t_all.append(x_t.clone())
        if return_all:
            return x_t, torch.stack(x_t_all, dim=0)
        else:
            return x_t



class DSB(BaseModel):
    def __init__(self, *args, direction=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.direction = direction
        self.num_timesteps = self.noiser.training_timesteps

        self.noiser.prepare_gamma_dsb()
        self.gammas = self.noiser.gammas

    def get_coef_ts(self, x, t, delta=1):
        #print(self.noiser.coefficient(t))
        coef_t = check_size(x, self.noiser.coefficient(t))
        coef_t_other = check_size(x, self.noiser.coefficient(t + delta))
        return coef_t, coef_t_other


    def _forward(self, x, t):
        x_other = self.forward(x, t)
        #print('forward shape :', x.shape, x_other.shape)
        if self.args.reparam == 'flow':
            v_pred = x_other
            if self.direction == 'b':
                coef_t, coef_t_next = self.get_coef_ts(x, t, 1)
                x = x + (coef_t_next['coef1'] - coef_t['coef1']) * v_pred
            elif self.direction == 'f':
                coef_t, coef_t_next = self.get_coef_ts(x, self.num_timesteps - t, -1)
                x = x + (coef_t_next['coef0'] - coef_t['coef0']) * v_pred
        elif self.args.reparam == 'term':
            if self.direction == 'b':
                coef_t, coef_t_next = self.get_coef_ts(x, t, 1)
                x_1 = x_other
                x_0 = (x - coef_t['coef1'] * x_1) / coef_t['coef0']
            elif self.direction == 'f':
                coef_t, coef_t_next = self.get_coef_ts(x, self.num_timesteps - t, -1)
                x_0 = x_other
                x_1 = (x - coef_t['coef0'] * x_0) / coef_t['coef1']
            x = coef_t_next['coef0'] * x_0 + coef_t_next['coef1'] * x_1
        else:
            t = t.long()
            std_t = self.s[t].view(shape)
            std_tm1 = self.s[t-1].view(shape)
            mu_x0_t = self.mu_x0[t].view(shape)
            mu_x0_tm1 = self.mu_x0[t-1].view(shape)
            mu_y_t = self.mu_y[t].view(shape)
            mu_y_tm1 = self.mu_y[t-1].view(shape)

            var_t = std_t**2
            var_tm1 = std_tm1**2
            var_t_tm1 = var_t - var_tm1 * (mu_x0_t / mu_x0_tm1)**2
            v = var_t_tm1 * (var_tm1 / var_t)

            x_tm1_mean = mu_x0_tm1 * x0 + mu_y_tm1 * y + \
                ((var_tm1 - v) / var_t).sqrt() * (x_t - mu_x0_t * x0 - mu_y_t * y)

            x = x_tm1_mean + v.sqrt() * torch.randn_like(x_t)
        return x
    
    def q_sample(self, t, x0, y):
        """ Sample q(x_t | x_0, y) """
        shape = [-1] + [1] * (x0.ndim - 1)

        mu_x0 = self.noiser.mu_x0[t].view(shape)
        mu_y = self.noiser.mu_y[t].view(shape)
        std = self.noiser.std[t].view(shape)

        x_t = mu_x0*x0 + mu_y*y + std*torch.randn_like(x0)
        
        return x_t.detach()

    def q_posterior(self, t, x_t, x0, y):
        """ Sample p(x_{t-1} | x_t, x0, y) """
        shape = [-1] + [1] * (x0.ndim - 1)
        #print(shape, self.s, t)
        #t = t.long()
        std_t = self.noiser.s[t].view(shape)
        std_tm1 = self.noiser.s[t-1].view(shape)
        mu_x0_t = self.noiser.mu_x0[t].view(shape)
        mu_x0_tm1 = self.noiser.mu_x0[t-1].view(shape)
        mu_y_t = self.noiser.mu_y[t].view(shape)
        mu_y_tm1 = self.noiser.mu_y[t-1].view(shape)

        var_t = std_t**2
        var_tm1 = std_tm1**2
        var_t_tm1 = var_t - var_tm1 * (mu_x0_t / mu_x0_tm1)**2
        v = var_t_tm1 * (var_tm1 / var_t)

        #print(mu_x0_tm1, mu_y_tm1, var_tm1, var_t, mu_x0_t, mu_y_t)
        
        x_tm1_mean = mu_x0_tm1 * x0 + mu_y_tm1 * y + \
            ((var_tm1 - v) / var_t).sqrt() * (x_t - mu_x0_t * x0 - mu_y_t * y)

        #print(x_tm1_mean, v.sqrt())
        x_tm1 = x_tm1_mean + v.sqrt() * torch.randn_like(x_t)

        return x_tm1

    def inference(self, x, sample=False):
        ones = torch.ones(size=(x.shape[0],), dtype=torch.int64, device=self.device)
        x_cache, gt_cache, t_cache, x0 = [], [], [],[]
        
        #输入随机加噪
        timesteps = torch.arange(self.num_timesteps, 0, -1, device=self.device)
        timesteps = timesteps.unsqueeze(1).repeat(1, x.shape[0])
        x_t = self.q_sample(timesteps[0], torch.zeros_like(x), x)
        x_t = x_t.float()
        x = x.float()
        #print('x:', x.max(), x.min())
        x_raw = x.clone()
        #print(x.min(), x.max(), x_t.min(), x_t.max())
        
        with torch.no_grad():
            for t in timesteps:
                x0_r = torch.zeros_like(x_t).to(self.device)
                x0_r = x0_r.float()
                tt = t
                #print(tt)
                with torch.autocast(device_type="cuda", enabled=self.args.use_amp):
                    x_input = torch.cat((x_t, x0_r), axis=1)
                    x_other = self.forward(x_input, tt, x0_r)
                    #print('model output:', x_other.max(), x_other.min())
                    x_cache.append(x_t.clone())
                #print('input:', tt, x_t.max(), x_t.min(), x_other.max(), x_other.min(), x.max(), x.min())
                x_tm1_pred = self.q_posterior(t, x_t, x_other, x)
                #print('q_posterior', x_tm1_pred.max(), x_tm1_pred.min())
            
                t_cache.append(self.num_timesteps + 1 - tt)
                x_t = x_tm1_pred
                gt_cache.append(x_raw)
            
        for i in range(len(timesteps)):
            x0.append(x_tm1_pred)
        
        #x_cache.reverse()
        x_cache = torch.stack(x_cache+ [x_other], dim=0).cpu() if sample else torch.cat(x_cache, dim=0).cpu()
        gt_cache = torch.cat(gt_cache, dim=0).cpu()
        t_cache = torch.cat(t_cache, dim=0).cpu()
        x0 = torch.cat(x0, dim=0).cpu()
        #print(x_cache.shape, gt_cache.shape)
        if sample:
            return x_cache, x0, t_cache, gt_cache
        else:
            return x_cache, gt_cache, t_cache, x0

def create_model(name, *args, **kwargs):
    name = name.lower()
    if 'diffusion' in name:
        model = Diffusion
    elif 'flow' in name:
        model = Flow
    elif 'dsb' in name:
        model = DSB
    else:
        raise NotImplementedError
    model = model(*args, **kwargs)
    return model
