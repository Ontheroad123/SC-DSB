import torch, os, tqdm, math, os.path as osp
import sys
from torchvision.utils import save_image
from torch.nn import functional as F

from util.dataset_us_ct_pair import create_data
#from util.dataset_us_ct_pair_LiTS2US import create_data
from util.noiser import create_noiser
from util.model import create_model
#from util.model_nosource import create_model
from util.logger import Logger
from util.visualize_us_ct_allmetrics import InferenceResultVisualizer, TrajectoryVisualizer
from util.vgg_perceptual_loss_png import VGGPerceptualLoss
from util.discriminator import Discriminator_large

class Runner():
    def __init__(self, args):
        self.args = args
        self.rank = self.args.local_rank

        self.dsb = 'dsb' in self.args.method

        self.device = torch.device("cuda")
        print('use device:',self.device)
        base_steps_per_epoch = 2**10 


        self.prior_set, self.prior_sampler, self.prior_loader = create_data(
            self.args.prior, self.args.gpus, dataset_size=base_steps_per_epoch*self.args.batch_size, 
            batch_size=self.args.batch_size)
        self.data_set, self.data_sampler, self.data_loader = create_data(
            self.args.dataset, self.args.gpus, dataset_size=base_steps_per_epoch*self.args.batch_size, 
            batch_size=self.args.batch_size)
        self.prior_iterator, self.data_iterator = iter(self.prior_loader), iter(self.data_loader)

        self.criterion = torch.nn.MSELoss(reduction='none')
        self.perceptual_loss = VGGPerceptualLoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)

        if not self.args.exp2d:
            self.val_prior_set = None
            self.val_data_set = None
            if self.args.val_prior is not None and self.args.val_dataset is not None:
                #验证集不需要扩大，原始大小
                self.val_prior_set, _, self.val_prior_loader = create_data(
                    self.args.val_prior, self.args.gpus, dataset_size=-1, 
                    batch_size=self.args.batch_size)
                self.val_data_set, _,  self.val_data_loader = create_data(
                    self.args.val_dataset, self.args.gpus, dataset_size=-1, 
                    batch_size=self.args.batch_size)
                self.val_prior_iterator, self.val_data_iterator = iter(self.val_prior_loader), iter(self.val_data_loader)

        if self.dsb:
            assert self.args.training_timesteps == self.args.inference_timesteps

            self.noiser = create_noiser(self.args.noiser, args, self.device)

            self.cache_size = self.cnt = base_steps_per_epoch * self.args.batch_size * 4

            self.backward_model = create_model(self.args.method, self.args, self.device, self.noiser, rank=self.rank, direction='b')
            self.forward_model = create_model(self.args.method, self.args, self.device, self.noiser, rank=self.rank, direction='f')

            if self.rank == 0:
                print(f'Backward Network #Params: {sum([p.numel() for p in self.backward_model.parameters()])}')
                print(f'Forward Network #Params: {sum([p.numel() for p in self.forward_model.parameters()])}')

            self.backward_model = self.backward_model.to(self.device)
            self.forward_model = self.forward_model.to(self.device)

            self.backward_optimizer = torch.optim.AdamW(
                self.backward_model.parameters(), 
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
            self.forward_optimizer = torch.optim.AdamW(
                self.forward_model.parameters(), 
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )

            self.model = {'backward': self.backward_model, 'forward': self.forward_model}
            self.optimizer = {'backward': self.backward_optimizer, 'forward': self.forward_optimizer}
            self.direction = 'backward'
        else:
            self.noiser = create_noiser(self.args.noiser, args, self.device)
            self.cache_size = self.cnt = base_steps_per_epoch * self.args.batch_size * 4
            self.model = create_model(self.args.method, self.args, self.device, self.noiser, rank=self.rank)
            if self.rank == 0:
                print(f'#Params: {sum([p.numel() for p in self.model.parameters()])}')

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        #Discriminator_large
        self.discriminator = Discriminator_large().to(self.device)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001 , betas=[0.5, 0.9])
        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.scheduler_d = CosineAnnealingLR(self.optimizer_d, T_max=50, eta_min=1e-5)

        self.load_ckpt()

        self.save_path = os.path.join('exp', self.args.exp_name)
        print(self.args.global_rank)
        if self.args.global_rank == 0:
            self.evaluators = {
                'Inference': InferenceResultVisualizer(self.args, self.device, save_path=self.save_path),
                'Trajectory': TrajectoryVisualizer(self.args, self.device, save_path=self.save_path),
            }
            self.logger = Logger(os.path.join(self.save_path, 'log'), self.noiser.training_timesteps)
            #self.logger.info(f"training command: {sys.argv}")
       
    def _next_batch(self):
        try:
            x_0, x_1 = next(self.prior_iterator), next(self.data_iterator)
        except StopIteration:
            self.prior_iterator, self.data_iterator = iter(self.prior_loader), iter(self.data_loader)
            x_0, x_1 = next(self.prior_iterator), next(self.data_iterator)
        x_0, x_1 = x_0.to(self.device), x_1.to(self.device)
        return x_0, x_1

    def next_batch(self, epoch, dsb=False):
        
        if self.cnt + self.args.batch_size > self.cache_size:
            self.x_cache, self.gt_cache, self.t_cache, self.x0_cache = [], [], [], []
            num_cache = math.ceil(self.cache_size / self.args.batch_size / self.noiser.num_timesteps)
            pbar = tqdm.trange(num_cache, desc=f'Cache epoch {epoch} model {self.direction}') if self.rank == 0 else range(num_cache)
            for _ in pbar:
                x_0, x_1 = self._next_batch()
                with torch.no_grad():
                    if self.direction == 'backward' :
                        _x_cache, _gt_cache, _t_cache, x0_cache = self.noiser.trajectory(x_0, x_1)
                    else:
                        model = self.model['backward' if self.direction == 'forward' else 'forward']
                        model.eval()
                        _x_cache, _gt_cache, _t_cache, x0_cache = model.inference(x_1 if self.direction == 'forward' else x_0)

                self.x_cache.append(_x_cache)
                self.gt_cache.append(_gt_cache)
                self.t_cache.append(_t_cache)
                self.x0_cache.append(x0_cache)
            self.x_cache = torch.cat(self.x_cache, dim=0).cpu()
            self.gt_cache = torch.cat(self.gt_cache, dim=0).cpu()
            self.t_cache = torch.cat(self.t_cache, dim=0).cpu()
            self.x0_cache = torch.cat(self.x0_cache, dim=0).cpu()
            self.cnt = 0
            self.indexs = torch.randperm(self.x_cache.shape[0])

        index = self.indexs[self.cnt:self.cnt + self.args.batch_size]
        self.cnt += self.args.batch_size
        x = self.x_cache[index]
        gt = self.gt_cache[index]
        t = self.t_cache[index]
        x0_input = self.x0_cache[index]
        x, gt, t, x0_input = x.to(self.device), gt.to(self.device), t.to(self.device), x0_input.to(self.device)
        
        return x, gt, t, x0_input

    def train(self):
        steps_per_epoch = int(len(self.data_loader) * self.args.repeat_per_epoch)
        self.cache_size = min(self.cache_size, steps_per_epoch * self.args.batch_size)
        for epoch in range(self.args.epochs):
 
            if self.dsb:
                self.cnt = self.cache_size
                self.direction = 'backward' if epoch % 2 == 0 else 'forward'
                model, optimizer = self.model[self.direction], self.optimizer[self.direction]
            else:
                model, optimizer = self.model, self.optimizer

            if epoch < self.args.skip_epochs:
                print(f'Skipping ep{epoch} and evaluate.')
                _,_ = self.evaluate(epoch, 0, last=True)
                continue
            self.noiser.train()
            
            model.train()
            pbar = tqdm.tqdm(total=steps_per_epoch)
            ema_loss, ema_loss_w = None, lambda x: min(0.99, x / 10)
            if self.prior_sampler is not None:
                self.prior_sampler.set_epoch(epoch)
                self.data_sampler.set_epoch(epoch)

            for i in range(steps_per_epoch):
                if self.dsb:
                    x_t, gt, t, x_1 = self.next_batch(epoch, dsb=True)
                else:
                    x_0, x_1, t = self.next_batch(epoch)
                    x_t = self.noiser(x_0, x_1, t)
                    gt = model.target(x_0, x_1, x_t, t)

                optimizer.zero_grad()
                optimizer_d.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.args.use_amp):
                    
                    # Part 1.a: Train discriminator with real data
                    x_t1 = x_t.float()
                    gt1 = gt.float()
                    disc_out = self.discriminator(x_t1, gt1, t)
                    real_loss = self.adversarial_loss(disc_out, is_real=True)

                    x0_r = torch.zeros_like(x_t).to(self.device)
                    x0_r = x0_r.float()

                    x_1 = x_1.to(self.device)
                    x = torch.cat((x_t.detach(), x_1), axis=1)

                    pred = model(x, t, x0_r)

                    # Part 1.b: Train discriminator with fake data
                    pred1 = pred.float()
                    disc_out = self.discriminator(pred1, x_t1, t)
                    fake_loss = self.adversarial_loss(disc_out, is_real=False)
                
                    # Compute total loss
                    d_loss = real_loss + fake_loss
                self.scaler.scale(d_loss).backward()
                self.scaler.step(optimizer_d)
                self.scaler.update()

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                   
                    gt = gt.float()
                    x_t = x_t.float()
                    x_1 = x_1.float()

                    x0_r = torch.zeros_like(x_t).to(self.device)
                    x0_r = x0_r.float()

                    x_1 = x_1.to(self.device)
                    x = torch.cat((x_t.detach(), x_1), axis=1)
                    x0_pred = model(x, t, x0_r)
                                        
                    #l1
                    rec_loss = F.l1_loss(x0_pred, gt, reduction="sum")

                    #l2
                    #mseloss = torch.nn.MSELoss(reduction='mean')
                    #rec_loss = mseloss(x0_pred.float(), gt.float()).mean()

                    #Compute adversarial loss
                    Posterior sampling q(x_{t-1} | x_t, y, x0_pred)
                    x_tm1_pred = model.q_posterior(t, x_t, x0_pred, x_1)
                    adv_loss = self.adversarial_loss(
                        self.discriminator(x_tm1_pred, x_t, t), is_real=True)

                    #perceptual_loss
                    perceptual_loss = self.perceptual_loss(x0_pred, gt, x_t).mean(dim=-1)

                    raw_loss = rec_loss.float()+perceptual_loss+adv_loss
                    loss = raw_loss
                    
                ema_loss = loss.item() if ema_loss is None else (ema_loss * ema_loss_w(i) + loss.item() * (1 - ema_loss_w(i)))
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                if  i == steps_per_epoch - 1:
                    if self.args.global_rank == 0:
                        self.logger.log_step(i, epoch * steps_per_epoch + i, epoch, ema_loss, raw_loss, t, dsb=self.dsb, direction=self.direction if self.dsb else '')
                    desc = f'epoch {epoch}, direction {self.direction}, iteration {i}, loss {ema_loss:.14f}' if self.dsb else f'epoch {epoch}, iteration {i}, loss {ema_loss:.14f}'
                    pbar.set_description(desc, refresh=False)
                    pbar.update(i + 1 - pbar.n)

            metrics_p, metrics_q = self.evaluate(epoch, i + 1, last=True)
            metrics = {}
            if self.args.global_rank == 0:
                if len(metrics_p.keys())>0 :
                    if self.direction =='backward':
                        metrics = metrics_q
                    else:
                        metrics = metrics_p
                else:
                    metrics = metrics_q
                self.logger.log_metrics(i, epoch * steps_per_epoch + i, epoch, metrics, dsb=self.dsb, direction=self.direction if self.dsb else '')
               
            
    def evaluate(self, epoch, iters, last=False):
        metrics_p, metrics_q = {}, {}
        mets = ['psnrs', 'ssims','fid','kid','lpips','kl', 'loss_v']
        for x in mets:
            metrics_p[x] = []
            metrics_q[x] = []

        with torch.no_grad():

            if self.dsb :
                self.backward_model.eval()
                self.forward_model.eval()
            else:
                self.model.eval()

            print('global_rank:', self.args.global_rank)
            if self.args.global_rank == 0:
                if last:
                    self.save_ckpt(epoch, iters)
                for i in range(len(self.val_prior_loader)):
                    if not self.args.exp2d:
                        try:
                            x_prior, x_data = next(self.val_prior_iterator), next(self.val_data_iterator)
                        except StopIteration:
                            self.val_prior_iterator, self.val_data_iterator = iter(self.val_prior_loader), iter(self.val_data_loader)
                            x_prior, x_data = next(self.val_prior_iterator), next(self.val_data_iterator)
                        x_prior, x_data = x_prior.to(self.device), x_data.to(self.device)
                    else:
                        x_prior, x_data, _ = self._next_batch(epoch)

                    if self.dsb:
                        qs, qgt, _, qx0 = self.backward_model.inference(x_data,  sample=True)
                        if epoch == 0:
                            ps, pgt, _, px0 = self.noiser.trajectory(x_prior, x_data, sample=True)
                        else:
                            ps, pgt, _, px0  = self.forward_model.inference(x_prior, sample=True)
                    else:
                        qs = self.model.inference(x_prior, return_all=True)[1]
                        ps = self.noiser.trajectory(x_prior, x_data)
                   
                    x_prior, x_data, qs, ps = x_prior.to(self.device), x_data.to(self.device), qs.to(self.device), ps.to(self.device)

                    if self.dsb:
                        t_metrics_q = self.evaluators['Inference'].draw(epoch, i, qs[-1], x_prior, x_data, subfix=f'_q')
                        t_metrics_p = self.evaluators['Inference'].draw(epoch, i, ps[-1], x_data, x_prior, subfix=f'_p')
                    else:
                        t_metrics_q = self.evaluators['Inference'].draw(epoch, i, qs[-1], x_data, x_prior)

                    for x in mets:
                        metrics_p[x].append(t_metrics_p[x])
                        metrics_q[x].append(t_metrics_q[x])

                    if last:
                        self.evaluators['Trajectory'].draw(epoch, i, xs=qs, subfix='_q')
                        self.evaluators['Trajectory'].draw(epoch, i, xs=ps, subfix='_p')
                    
        return metrics_p, metrics_q

    def save_ckpt(self, epoch, iters):
        os.makedirs(os.path.join(self.save_path, 'ckpt'), exist_ok=True)
        if self.dsb:
            ckpt = {
                'backward_model': self.backward_model.state_dict(),
                'forward_model': self.forward_model.state_dict(),
                'backward_optimizer': self.backward_optimizer.state_dict(),
                'forward_optimizer': self.forward_optimizer.state_dict(),
            }
        else:
            ckpt = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
        torch.save(ckpt, os.path.join(self.save_path, 'ckpt', f'ep{epoch}_it{iters}.pth'))

    def load_ckpt(self):
        def match_ckpt(ckpt):
            _ckpt = {}
            for k, v in ckpt.items():
              if 'generator' in k:
                newk = k.replace('generator', 'network')
                _ckpt[newk] = v
            return _ckpt
        if self.args.ckpt is not None:
            ckpt = torch.load(self.args.ckpt, map_location='cpu')
            if self.dsb:
                self.backward_model.load_state_dict(match_ckpt(ckpt['backward_model']), strict=False)
                self.forward_model.load_state_dict(match_ckpt(ckpt['forward_model']), strict=False)
                if "backward_optimizer" in ckpt:
                    self.backward_optimizer.load_state_dict(ckpt['backward_optimizer'])
                    self.forward_optimizer.load_state_dict(ckpt['forward_optimizer'])
            else:
                self.model.load_state_dict(match_ckpt(ckpt['model']), strict=False)
                if "optimizer" in ckpt:
                    self.optimizer.load_state_dict(ckpt['optimizer'])
        else:
            if self.args.backward_ckpt is not None:
                ckpt = torch.load(self.args.backward_ckpt, map_location='cpu')
                self.backward_model.load_state_dict(match_ckpt(ckpt['backward_model']), strict=False)
                #self.backward_model.load_state_dict(match_ckpt(ckpt['state_dict']), strict=True)
                print('backward_model load ckpt:', self.args.backward_ckpt)
            if self.args.forward_ckpt is not None:
                ckpt = torch.load(self.args.forward_ckpt, map_location='cpu')
                self.forward_model.load_state_dict(match_ckpt(ckpt['forward_model']), strict=False)
                #self.forward_model.load_state_dict(match_ckpt(ckpt['state_dict']), strict=True)
                print('forward_model load ckpt:', self.args.forward_ckpt)