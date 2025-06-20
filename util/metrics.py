import os
import warnings
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
import matplotlib.pyplot as plt

import torchvision.models as models
from scipy.linalg import sqrtm
from torch.nn.functional import adaptive_avg_pool2d
from scipy.stats import entropy
from sklearn.metrics.pairwise import rbf_kernel
import lpips 
from skimage import io, metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cal_kid(real_images, generated_images):
    """
    计算 KID 指标。
    
    Args:
        images1: 第一个图像数据集，形状为 (N, C, H, W)。
        images2: 第二个图像数据集，形状为 (N, C, H, W)。
        model: 预训练的 Inception-v3 模型。

    Returns:
        KID 指标的值。
    """
    
    # 加载预训练的 Inception-v3 模型
    model = models.inception_v3(pretrained=True).to(device)
    model.eval()  # 设置模型为评估模式

    # 将图像数据转换为 PyTorch 张量
    images1 = torch.from_numpy(real_images)
    images2 = torch.from_numpy(generated_images)
    images1 = images1.to(device).to(torch.float)
    images2 = images2.to(device).to(torch.float)
    # 将单通道图像扩展为三通道图像
    
    images1 = torch.tile(images1, (1, 3, 1, 1)) 
    images2 = torch.tile(images2, (1, 3, 1, 1)) 
    #print(images1.shape, images2.shape)
    # 使用 Inception-v3 模型提取特征
    act1 = model(images1)
    act2 = model(images2)
    # 计算核矩阵
    kernel = rbf_kernel(act1.cpu().detach().numpy(), act2.cpu().detach().numpy())
    
    # 计算 KID 指标
    kid = entropy(kernel.mean(axis=1), base=2) - entropy(kernel, base=2).mean()
    return kid

from torchmetrics.image.fid import FrechetInceptionDistance
def cal_fid(real_images, generated_images):
    fid = FrechetInceptionDistance(feature=2048).to(device)
    fid_score = 0
    real_images = torch.Tensor(real_images).to(torch.uint8).to(device) 
    generated_images = torch.Tensor(generated_images).to(torch.uint8).to(device) 
    real_images = torch.tile(real_images, (1, 3, 1, 1)).to(device)
    generated_images = torch.tile(generated_images, (1, 3, 1, 1)).to(device) 
    print('fid shape:', real_images.shape, generated_images.shape)
    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)
    fid_score += fid.compute()
    #print(f"FID: {fid_score/real_images.shape[0]}")
    return fid_score/real_images.shape[0]

def cal_lpips(images1, images2):
    lpips_model = lpips.LPIPS(net='alex', version='0.1') 
    lpips_model.to(device)  # 将模型移动到设备 (CPU 或 GPU)
    lpips_model.eval()  # 设置模型为评估模式
    # 将图像数据转换为 PyTorch 张量
    #images1 = torch.from_numpy(images1).to(device)
    #images2 = torch.from_numpy(images2).to(device)

    # 将图像数据转换为 PyTorch 张量
    images1 = torch.from_numpy(images1)
    images2 = torch.from_numpy(images2)
    images1 = images1.to(device).to(torch.float)
    images2 = images2.to(device).to(torch.float)
    
    # 将单通道图像扩展为三通道图像
    # images1 = torch.tile(images1, (1, 3, 1, 1)) 
    # images2 = torch.tile(images2, (1, 3, 1, 1)) 
    
    lpips_distance = lpips_model(images1, images2)
    #print(lpips_distance.mean())
    return lpips_distance.mean()

def cal_kl(images1, images2):
    # 计算直方图
    hist1, _ = np.histogram(images1.flatten(), bins=256, density=True) # 使用 density=True 进行归一化
    hist2, _ = np.histogram(images2.flatten(), bins=256, density=True)

    #  避免 0 值， 对 q 添加一个很小的数值
    hist2 = hist2 + 1e-8
    # 计算 KL 散度
    kl_div = entropy(hist1, hist2) # 使用 scipy 计算 KL 散度
    return kl_div


def save_image_pair(x0, x0_pred, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    n_image = min(4, x0.shape[0])
    fig, axes = plt.subplots(nrows=2, ncols=n_image, figsize=(n_image*2, 4))

    if n_image == 1:
        axes = axes[..., None]

    for i in range(n_image):
        axes[0, i].imshow(x0[i].permute(1, 2, 0).cpu().numpy(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(x0_pred[i].permute(1, 2, 0).cpu().numpy(), cmap='gray')
        axes[1, i].axis('off')

    plt.tight_layout(pad=0.1)
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()


def save_eval_images(
    source_images,
    target_images,
    pred_images,
    psnrs,
    ssims,
    save_path
):
    h, w = 30, 30
    zoom_region = [100-w, 100+w, 100-h, 100+h]
    zoom_size = [0, -0.4, 1, 0.47]

    # Squeeze channel dimension
    source_images = source_images.squeeze()
    target_images = target_images.squeeze()
    pred_images = pred_images.squeeze()

    # If images between [-1, 1], scale to [0, 1]
    if np.nanmin(source_images) < -0.1:
        source_images = ((source_images + 1) / 2).clip(0, 1)

    if np.nanmin(target_images) < -0.1:
        target_images = ((target_images + 1) / 2).clip(0, 1)

    if np.nanmin(pred_images) < -0.1:
        pred_images = ((pred_images + 1) / 2).clip(0, 1)
    
    plt.style.use('dark_background')

    for i in range(len(source_images)):
        fig, ax = plt.subplots(1, 3, figsize=(12*1.5,8*1.5))
        
        ax_zoomed(ax[0], mean_norm(source_images[i]), zoom_region, zoom_size)
        ax_zoomed(ax[1], mean_norm(target_images[i]), zoom_region, zoom_size)
        ax_zoomed(ax[2], mean_norm(pred_images[i]), zoom_region, zoom_size)
        
        ax[0].set_title('Source')
        ax[1].set_title('Target')
        ax[2].set_title(f'PSNR: {psnrs[i]:.2f}\nSSIM: {ssims[i]:.2f}')

        # Save figure
        path = os.path.join(save_path, 'sample_images', f'slice_{i}.png')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)


def save_preds(preds, path):
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds)

    # Normalize predictions
    preds = ((preds + 1) / 2).clip(0, 1)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, preds)


def to_norm(x):
    x = x/2
    x = x + 0.5
    return x.clip(0, 1)

def norm_01(x):
    return (x - x.min(axis=(-1,-2), keepdims=True))/(x.max(axis=(-1,-2), keepdims=True) - x.min(axis=(-1,-2), keepdims=True))


def mean_norm(x):
    x = np.abs(x)
    return x/x.mean(axis=(-1,-2), keepdims=True)


def apply_mask_and_norm(x, mask, norm_func):
    x = x*mask
    x = norm_func(x)
    return x


def center_crop(x, crop):
    h, w = x.shape[-2:]
    x = x[..., h//2-crop[0]//2:h//2+crop[0]//2, w//2-crop[1]//2:w//2+crop[1]//2]
    return x


def ax_zoomed(
    ax,
    im,
    zoom_region,
    zoom_size,
    zoom_edge_color='yellow'
):
    ax.imshow(np.flip(im, axis=0), origin='lower', cmap='gray')
    x1, x2, y1, y2 = zoom_region
    axins = ax.inset_axes(
        zoom_size,
        xlim=(x1, x2), ylim=(y1, y2))
    
    axins.imshow(np.flip(im, axis=0), cmap='gray')

    # Add border to zoomed region
    for spine in axins.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(2)
    
    # Remove inset axes ticks/labels
    axins.set_xticks([])
    axins.set_yticks([])
    
    ax.indicate_inset_zoom(axins, edgecolor=zoom_edge_color, linewidth=3)
    ax.axis('off')


def compute_metrics(
    gt_images,
    pred_images, 
    mask=None,
    norm='mean',
    subject_ids=None,
    report_path=None
):
    """ Compute PSNR and SSIM between gt_images and pred_images.
    
    Args:
        gt_images (torch.Tensor): Ground truth images.
        pred_images (torch.Tensor): Predicted images.
        mask (torch.Tensor): Mask to apply to images.
        crop (tuple): Center crop images to (Height, Width).
        norm (str): Normalization method. Options: 'mean', '01'.
        subject_ids (list): List of subject IDs for each slice.

    Returns:
        dict: Dictionary containing PSNR and SSIM values.
    
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Compute psnr and ssim
        psnr_values = []
        ssim_values = []

        # Normalize function
        if norm == 'mean':
            norm_func = mean_norm
        elif norm == '01':
            norm_func = norm_01

        # If torch tensor, convert to numpy
        if isinstance(gt_images, torch.Tensor):
            gt_images = gt_images.cpu().numpy()

        if isinstance(pred_images, torch.Tensor):
            pred_images = pred_images.cpu().numpy()

        # If images between [-1, 1], scale to [0, 1]
        if np.nanmin(gt_images) < -0.1:
            gt_images = ((gt_images + 1) / 2).clip(0, 1)

        if np.nanmin(pred_images) < -0.1:
            pred_images = ((pred_images + 1) / 2).clip(0, 1)

        # Apply mask and normalize
        if mask is not None:
            # Crop to mask shape
            gt_images = center_crop(gt_images, mask.shape[-2:])
            pred_images = center_crop(pred_images, mask.shape[-2:])

            gt_images = apply_mask_and_norm(gt_images, mask, norm_func)
            pred_images = apply_mask_and_norm(pred_images, mask, norm_func)
        else:
            gt_images = norm_func(gt_images)
            pred_images = norm_func(pred_images)

        # Compute psnr and ssim
        for gt, pred in zip(gt_images, pred_images):
            gt = gt.squeeze()
            pred = pred.squeeze()
            
            psnr_value = psnr(gt, pred, data_range=gt.max())
            psnr_values.append(psnr_value)

            ssim_value = ssim(gt, pred, channel_axis=0, data_range=gt.max())
            ssim_values.append(ssim_value)

        fid = cal_fid(gt_images, pred_images)
        kid = cal_kid(gt_images, pred_images)
        lpips = cal_lpips(gt_images, pred_images)
        kl = cal_kl(gt_images, pred_images)

        fid_values=fid.cpu()
        kid_values=kid
        lpips_values=lpips.cpu()
        kl_values = kl

        # Convert list to numpy array
        psnr_values = np.asarray(psnr_values)
        ssim_values = np.asarray(ssim_values)
        fid_values = np.asarray(fid_values)
        kid_values = np.asarray(kid_values)
        lpips_values = np.asarray(lpips_values.detach().numpy())
        kl_values = np.asarray(kl_values)
        #print('psnr_values:', psnr_values)
        # Compute subject reports
        subject_reports = {}
        if subject_ids is not None:
            for i in np.unique(subject_ids):
                idx = np.where(subject_ids == i)[0]
                subject_report = {
                    'psnrs': psnr_values[idx],
                    'ssims': ssim_values[idx],
                    'fids': fid_values[idx],
                    'kids': kid_values[idx],
                    'lpips': lpips_values[idx],
                    'kl': kl_values[idx],

                    'psnr_mean': np.nanmean(psnr_values[idx]),
                    'ssim_mean': np.nanmean(ssim_values[idx]),
                    'fid_mean': np.nanmean(fid_values[idx]),
                    'kid_mean': np.nanmean(kid_values[idx]),
                    'lpips_mean': np.nanmean(lpips_values[idx]),
                    'kl_mean': np.nanmean(kl_values[idx]),

                    'psnr_std': np.nanstd(psnr_values[idx]),
                    'ssim_std': np.nanstd(ssim_values[idx]),
                    'fid_std': np.nanstd(fid_values[idx]),
                    'kid_std': np.nanstd(kid_values[idx]),
                    'lpips_std': np.nanstd(lpips_values[idx]),
                    'kl_std': np.nanstd(kl_values[idx]),
                }
                subject_reports[i] = subject_report
            
        # Compute mean and std values
        if subject_ids is not None:
            psnr_mean = np.nanmean([report['psnr_mean'] for report in subject_reports.values()])
            ssim_mean = np.nanmean([report['ssim_mean'] for report in subject_reports.values()])
            fid_mean = np.nanmean([report['fid_mean'] for report in subject_reports.values()])
            kid_mean = np.nanmean([report['kid_mean'] for report in subject_reports.values()])
            lpips_mean = np.nanmean([report['lpips_mean'] for report in subject_reports.values()])
            kl_mean = np.nanmean([report['kl_mean'] for report in subject_reports.values()])

            psnr_std = np.nanstd([report['psnr_std'] for report in subject_reports.values()])
            ssim_std = np.nanstd([report['ssim_std'] for report in subject_reports.values()])
            fid_std = np.nanstd([report['fid_std'] for report in subject_reports.values()])
            kid_std = np.nanstd([report['kid_std'] for report in subject_reports.values()])
            lpips_std = np.nanstd([report['lpips_std'] for report in subject_reports.values()])
            kl_std = np.nanstd([report['kl_std'] for report in subject_reports.values()])
        else:
            psnr_mean = np.nanmean(psnr_values)
            ssim_mean = np.nanmean(ssim_values)
            fid_mean = np.nanmean(fid_values)
            kid_mean = np.nanmean(kid_values)
            lpips_mean = np.nanmean(lpips_values)
            kl_mean = np.nanmean(kl_values)

            psnr_std = np.nanstd(psnr_values)
            ssim_std = np.nanstd(ssim_values)
            fid_std = np.nanstd(fid_values)
            kid_std = np.nanstd(kid_values)
            lpips_std = np.nanstd(lpips_values)
            kl_std = np.nanstd(kl_values)
        
        if report_path is not None:
            with open(report_path, 'w') as f:
                f.write(f'PSNR: {psnr_mean:.2f} ± {psnr_std:.2f}\n')
                f.write(f'SSIM: {ssim_mean:.2f} ± {ssim_std:.2f}\n')
                f.write(f'fid: {fid_mean:.2f} ± {fid_std:.2f}\n')
                f.write(f'kid: {kid_mean:.2f} ± {kid_std:.2f}\n')
                f.write(f'lpips: {lpips_mean:.2f} ± {lpips_std:.2f}\n')
                f.write(f'kl: {kl_mean:.2f} ± {kl_std:.2f}\n')
                f.write('\n')

                if subject_ids is not None:
                    for subject_id, report in subject_reports.items():
                        f.write(f'Subject {subject_id}\n')
                        f.write(f'PSNR: {report["psnr_mean"]:.2f} ± {report["psnr_std"]:.2f}\n')
                        f.write(f'SSIM: {report["ssim_mean"]:.2f} ± {report["ssim_std"]:.2f}\n')
                        f.write(f'fid: {report["fid_mean"]:.2f} ± {report["fid_std"]:.2f}\n')
                        f.write(f'kid: {report["kid_mean"]:.2f} ± {report["kid_std"]:.2f}\n')
                        f.write(f'lpips: {report["lpips_mean"]:.2f} ± {report["lpips_std"]:.2f}\n')
                        f.write(f'kl: {report["kl_mean"]:.2f} ± {report["kl_std"]:.2f}\n')
                        f.write('\n')         

        res = {
            # 'psnr_mean': psnr_mean,
            # 'ssim_mean': ssim_mean,
            # 'fid_mean': fid_mean,
            # 'kid_mean': kid_mean,
            # 'lpips_mean': lpips_mean,
            # 'kl_mean': kl_mean,

            # 'psnr_std': psnr_std,
            # 'ssim_std': ssim_std,
            # 'fid_std': fid_std,
            # 'kid_std': kid_std,
            # 'lpips_std': lpips_std,
            # 'kl_std': kl_std,

            'psnrs': psnr_values,
            'ssims': ssim_values,
            'fid': fid_values,
            'kid': kid_values,
            'lpips': lpips_values,
            'kl': kl_values,
            'subject_reports': subject_reports
        }

        return res


# import cv2
# import os, os.path as osp
# import numpy as np
# path = '/home/lzh/SDSB-main/dataset/ct-us_2022-2023_liver/us_val/00008711_20220502'
# reads = []
# preds = []
# for fil in os.listdir(path):
#     img = cv2.imread(osp.join(path, fil))/255
#     img = cv2.resize(img, (512,512))
#     reads.append(img )
# path = '/home/lzh/SDSB-main/dataset/ct-us_2022-2023_liver/us_val/00008711_20220502'
# for fil in os.listdir(path):
#     img = cv2.imread(osp.join(path, fil))/255
#     img = cv2.resize(img, (512,512))
#     preds.append(img )

# read = np.transpose(np.array(reads), (0,3,1,2))
# pred = np.transpose(np.array(preds), (0,3,1,2))
# print(read.shape, read.max())

# res = compute_metrics(read, pred)
# print(res)

# psnr_value = psnr(read, pred, data_range=read.max())
# fid = cal_fid(read, pred)
# kid = cal_kid(read, pred)
# lpips = cal_lpips(read, pred)
# kl = cal_kl(read, pred)

# ssim_values = []
# for gt, pred in zip(read, pred):
#     gt = gt.squeeze()
#     pred = pred.squeeze()
#     ssim_value = ssim(gt, pred, channel_axis=0, data_range=gt.max())*100
#     ssim_values.append(ssim_value)

# print('psnr:', psnr_value, ssim_values.mean(), 'fid:',fid, 'kid:',kid, 'lpips:',lpips, 'kl:', kl)