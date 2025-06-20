import os, tensorboardX, time, sys, logging, json, numpy as np


class Logger:
    def __init__(self, log_dir, training_timesteps, log_name='log', log_level=logging.INFO, log_console=False, log_file=True, log_tensorboard=True):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.training_timesteps = training_timesteps
        self.log_name = log_name
        self.log_level = log_level
        self.log_console = log_console
        self.log_file = log_file
        self.log_tensorboard = log_tensorboard
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        self.logger.handlers = []
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if log_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(self.formatter)
            self.logger.addHandler(console_handler)
        if log_file:
            file_handler = logging.FileHandler(os.path.join(log_dir, log_name + '.log'))
            file_handler.setLevel(log_level)
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)
        if log_tensorboard:
            self.tensorboard_writer = tensorboardX.SummaryWriter(os.path.join(log_dir, 'tensorboard'), flush_secs=30)

    def log(self, message, level=logging.INFO):
        self.logger.log(level, message)
        if self.log_tensorboard:
            self.tensorboard_writer.add_text('log', message, time.time())

    def log_dict(self, dict, level=logging.INFO):
        self.log(json.dumps(dict, indent=4), level)

    def log_image(self, tag, image, step):
        if self.log_tensorboard:
            self.tensorboard_writer.add_image(tag, image, step)

    def log_scalar(self, tag, value, step):
        if self.log_tensorboard:
            self.tensorboard_writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, values, step, bins=1000):
        if self.log_tensorboard:
            self.tensorboard_writer.add_histogram(tag, values, step, bins)

    def log_graph(self, model, input_to_model):
        if self.log_tensorboard:
            self.tensorboard_writer.add_graph(model, input_to_model)

    def log_text(self, tag, text, step):
        if self.log_tensorboard:
            self.tensorboard_writer.add_text(tag, text, step)

    def log_step(self, step, global_step, epoch, ema_loss, loss, timestep, dsb=False, direction=''):
        self.log_scalar('loss', ema_loss, global_step)
        self.log_scalar('log_loss', np.log10(ema_loss), global_step)
        if dsb:
            self.log_scalar(f'loss/ema/{direction}/ep{epoch}', ema_loss, step)
            self.log_scalar(f'log_loss/ema/{direction}/ep{epoch}', np.log10(ema_loss), step)
            self.log(f'epoch {epoch}, direction {direction}, iteration {step}, loss {ema_loss:.14f}')
        else:
            self.log_scalar(f'loss/ema/ep{epoch}', ema_loss, step)
            self.log_scalar(f'log_loss/ema/ep{epoch}', np.log10(ema_loss), step)
            self.log(f'epoch {epoch}, iteration {step}, loss {ema_loss:.14f}')

    def log_metrics(self, step, global_step, epoch, metrics, dsb=False, direction=''):
        ema_loss = np.mean(metrics['loss_v'])
        self.log_scalar('loss_v', ema_loss, global_step)
        #self.log_scalar('log_loss', np.log10(ema_loss), global_step)
        psnrs1 = []
        ssims1 = []
        fids1 = []
        kids1 = []
        lpips1 = []
        kls1 = []

        psnrs1 += list(metrics['psnrs'])
        ssims1 += list(metrics['ssims'])
        fids1.append(metrics['fid'])
        kids1.append(metrics['kid'])
        lpips1.append(metrics['lpips'])
        kls1.append(metrics['kl'])

        psnr_mean = np.nanmean(np.array(psnrs1))
        ssim_mean = np.nanmean(np.array(ssims1))
        fid_mean = np.nanmean(np.array(fids1))
        kid_mean = np.nanmean(np.array(kids1))
        lpips_mean = np.nanmean(np.array(lpips1))
        kl_mean = np.nanmean(np.array(kls1))

        psnr_std = np.nanstd(np.array(psnrs1))
        ssim_std = np.nanstd(np.array(ssims1))
        fid_std = np.nanstd(np.array(fids1))
        kid_std = np.nanstd(np.array(kids1))
        lpips_std = np.nanstd(np.array(lpips1))
        kl_std = np.nanstd(np.array(kls1))

        if dsb:
            self.log_scalar(f'loss_v/ema/{direction}/ep{epoch}', ema_loss, step)
            #self.log_scalar(f'log_loss/ema/{direction}/ep{epoch}', np.log10(ema_loss), step)
            #self.log(f'epoch {epoch}, direction {direction}, iteration {step}, loss_v {ema_loss:.14f}')
            self.log(f'epoch {epoch}, direction {direction}, iteration {step}, val_fid {fid_mean}')
            self.log(f'epoch {epoch}, direction {direction}, iteration {step}, val_kid {kid_mean}')
            self.log(f'epoch {epoch}, direction {direction}, iteration {step}, val_lpips {lpips_mean}')
            self.log(f'epoch {epoch}, direction {direction}, iteration {step}, val_psnr {psnr_mean}')
            self.log(f'epoch {epoch}, direction {direction}, iteration {step}, val_ssim {ssim_mean}')
            self.log(f'epoch {epoch}, direction {direction}, iteration {step}, val_kl {kl_mean}')
            self.log('\n')  
            self.log(f'epoch {epoch}, direction {direction}, iteration {step}, val_fid_std {fid_std}')
            self.log(f'epoch {epoch}, direction {direction}, iteration {step}, val_kid_std {kid_std}')
            self.log(f'epoch {epoch}, direction {direction}, iteration {step}, val_lpips_std {lpips_std}')
            self.log(f'epoch {epoch}, direction {direction}, iteration {step}, val_psnr_std {psnr_std}')
            self.log(f'epoch {epoch}, direction {direction}, iteration {step}, val_ssim_std {ssim_std}')
            self.log(f'epoch {epoch}, direction {direction}, iteration {step}, val_kl_std {kl_std}')


        else:
            self.log_scalar(f'loss_v/ema/ep{epoch}', ema_loss, step)
            self.log(f'epoch {epoch}, iteration {step}, val_psnr {np.mean(metrics["psnrs"])}')
            self.log(f'epoch {epoch}, iteration {step}, val_ssim {np.mean(metrics["ssims"])}')
            self.log(f'epoch {epoch}, iteration {step}, val_fid {np.mean(metrics["fid"])}')
            self.log(f'epoch {epoch}, iteration {step}, val_kid {np.mean(metrics["kid"])}')
            self.log(f'epoch {epoch}, iteration {step}, val_lpips {np.mean(metrics["lpips"])}')
            self.log(f'epoch {epoch}, iteration {step}, val_kl {np.mean(metrics["kl"])}')