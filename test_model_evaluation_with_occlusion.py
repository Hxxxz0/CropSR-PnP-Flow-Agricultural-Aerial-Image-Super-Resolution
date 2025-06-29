#!/usr/bin/env python3
"""
使用PnP-Flow方法测试带遮挡的CropSR超分辨率任务
测试在有随机方块遮挡情况下的图像恢复能力
基于预训练的Flow Matching模型进行条件图像恢复
"""

import torch
import numpy as np
import sys
import os
import argparse
from PIL import Image
import torch.nn.functional as F
from time import perf_counter
import random

# 添加项目路径
sys.path.append('.')
from pnpflow.utils import define_model, load_cfg_from_cfg_file, postprocess
from pnpflow.train_flow_matching import FLOW_MATCHING
from pnpflow.dataloaders import DataLoaders
from pnpflow.degradations import Superresolution
from pnpflow.methods.pnp_flow import PNP_FLOW
import pnpflow.utils as utils

class OccludedSuperresolution:
    """带遮挡的超分辨率降质类，避免创建巨大的降采样矩阵"""
    def __init__(self, sf, dim_image, device="cuda", occlusion_params=None):
        self.sf = sf
        self.device = device
        self.dim_image = dim_image
        self.occlusion_params = occlusion_params or {
            'num_blocks': 5,      # 遮挡方块数量
            'min_size': 20,       # 方块最小尺寸
            'max_size': 60,       # 方块最大尺寸
            'intensity': 0.0      # 遮挡强度 (0.0=黑色, 1.0=白色, 0.5=灰色)
        }
        
    def add_random_occlusion(self, x):
        """添加随机方块遮挡"""
        batch_size, channels, height, width = x.shape
        occluded_x = x.clone()
        
        # 为每个图像添加遮挡
        for b in range(batch_size):
            num_blocks = self.occlusion_params['num_blocks']
            min_size = self.occlusion_params['min_size']
            max_size = self.occlusion_params['max_size']
            intensity = self.occlusion_params['intensity']
            
            for _ in range(num_blocks):
                # 随机选择方块尺寸
                block_w = random.randint(min_size, min(max_size, width))
                block_h = random.randint(min_size, min(max_size, height))
                
                # 随机选择方块位置
                start_x = random.randint(0, width - block_w)
                start_y = random.randint(0, height - block_h)
                
                # 应用遮挡
                occluded_x[b, :, start_y:start_y+block_h, start_x:start_x+block_w] = intensity
        
        return occluded_x
    
    def H(self, x):
        """降质操作：HR -> 遮挡 -> LR，使用高效的插值方法"""
        # 先添加遮挡
        occluded_x = self.add_random_occlusion(x)
        # 使用插值进行降采样，避免巨大矩阵
        return F.interpolate(occluded_x, scale_factor=1/self.sf, mode='bilinear', align_corners=False)
    
    def H_adj(self, x):
        """上采样操作：LR -> HR，使用双三次插值"""
        return F.interpolate(x, scale_factor=self.sf, mode='bicubic', align_corners=False)

class Args:
    """配置类，模拟argparse参数"""
    def __init__(self, cfg):
        # 从配置文件复制基本参数
        for key, value in cfg.items():
            setattr(self, key, value)
        
        # PnP-Flow特定参数 (按照main.py中的设置)
        self.method = 'pnp_flow'
        self.problem = 'superresolution'  # 使用标准超分辨率问题
        self.noise_type = 'gaussian'
        self.sigma_noise = 0.05          # 按照main.py中gaussian噪声的标准设置
        self.max_batch = 8               # 测试样本数量
        self.batch_size_ip = 1           # 每次处理一张图像
        self.eval_split = 'test'
        
        # 从配置文件加载方法配置
        method_config_file = cfg.root + 'config/method_config/{}.yaml'.format(self.method)
        try:
            method_cfg = load_cfg_from_cfg_file(method_config_file)
            # 更新配置
            for key, value in method_cfg.items():
                setattr(self, key, value)
        except FileNotFoundError:
            print(f"警告: 未找到方法配置文件 {method_config_file}，使用默认参数")
                        # 默认PnP-Flow算法参数
            self.steps_pnp = 200
            self.lr_pnp = 1.5
            self.num_samples = 3
            self.gamma_style = 'alpha_1_minus_t'
            self.alpha = 1.0
        
        # 保存和计算选项
        self.save_results = True
        self.compute_time = False
        self.compute_memory = False
        
        # 创建方法配置字典
        self.dict_cfg_method = {}
        try:
            method_cfg = load_cfg_from_cfg_file(method_config_file)
            for key in method_cfg.keys():
                self.dict_cfg_method[key] = getattr(self, key)
        except:
            self.dict_cfg_method = {
                'steps_pnp': getattr(self, 'steps_pnp', 100),
                'lr_pnp': getattr(self, 'lr_pnp', 1.0),
                'num_samples': getattr(self, 'num_samples', 3),
                'gamma_style': getattr(self, 'gamma_style', 'constant')
            }

def clear_gpu_memory():
    """清理GPU显存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU显存已清理")

def load_model_and_config(model_path, device):
    """加载模型和配置"""
    print("=== 加载配置和模型 ===")
    
    # 加载配置
    cfg = load_cfg_from_cfg_file('./config/main_config.yaml')
    dataset_config = cfg.root + f'config/dataset_config/{cfg.dataset}.yaml'
    cfg.update(load_cfg_from_cfg_file(dataset_config))
    
    print(f"数据集: {cfg.dataset}")
    print(f"图像尺寸: {cfg.dim_image}x{cfg.dim_image}")
    print(f"使用设备: {device}")
    
    # 创建模型
    model, state = define_model(cfg)
    model = model.to(device)
    
    # 加载权重
    if os.path.exists(model_path):
        print(f"加载模型: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    else:
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    return model, cfg

def create_occluded_lr_images(hr_images, degradation, sigma_noise, device, seed=42):
    """创建带遮挡的LR图像"""
    print("=== 创建带遮挡的LR图像 ===")
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(seed)
    random.seed(seed)
    
    lr_images = []
    occluded_hr_images = []  # 保存遮挡后的HR图像用于可视化
    lr_masks = []           # 保存LR遮挡掩码
    
    for i, hr_img in enumerate(hr_images):
        hr_single = hr_img.unsqueeze(0).to(device)
        
        # 创建HR遮挡掩码
        mask_hr = torch.ones_like(hr_single[:, :1, :, :])  # 单通道掩码
        num_blocks = degradation.occlusion_params['num_blocks']
        min_size = degradation.occlusion_params['min_size']
        max_size = degradation.occlusion_params['max_size']
        intensity = degradation.occlusion_params['intensity']
        
        # 手动添加遮挡并同步更新掩码
        occluded_hr = hr_single.clone()
        _, _, H, W = hr_single.shape
        for _ in range(num_blocks):
            block_w = random.randint(min_size, min(max_size, W))
            block_h = random.randint(min_size, min(max_size, H))
            start_x = random.randint(0, W - block_w)
            start_y = random.randint(0, H - block_h)
            
            occluded_hr[:, :, start_y:start_y+block_h, start_x:start_x+block_w] = intensity
            mask_hr[:, :, start_y:start_y+block_h, start_x:start_x+block_w] = 0.0
        
        # HR->LR 降采样
        lr_single = F.interpolate(occluded_hr, scale_factor=1/degradation.sf, mode='bilinear', align_corners=False)
        lr_mask_single = F.interpolate(mask_hr, scale_factor=1/degradation.sf, mode='nearest')
        
        # 添加噪声
        if sigma_noise > 0:
            lr_single += torch.randn_like(lr_single) * sigma_noise
        
        lr_images.append(lr_single.cpu())
        occluded_hr_images.append(occluded_hr.cpu())
        lr_masks.append(lr_mask_single.cpu())
        
        print(f"创建第 {i+1} 张带遮挡的LR图像")
        print(f"  HR形状: {hr_single.shape}, 值域: [{hr_single.min():.3f}, {hr_single.max():.3f}]")
        print(f"  遮挡HR形状: {occluded_hr.shape}, 值域: [{occluded_hr.min():.3f}, {occluded_hr.max():.3f}]")
        print(f"  LR形状: {lr_single.shape}, 值域: [{lr_single.min():.3f}, {lr_single.max():.3f}]")
    
    return torch.cat(lr_images, dim=0), torch.cat(occluded_hr_images, dim=0), torch.cat(lr_masks, dim=0)

def save_occlusion_comparison_images(hr_images, occluded_hr_images, lr_images, restored_images, 
                                   bicubic_images, cfg, save_dir="./test_results_occlusion"):
    """保存遮挡对比图像：HR真实、遮挡HR、LR输入、双三次插值、PnP-Flow恢复"""
    print(f"=== 保存遮挡对比图像到 {save_dir} ===")
    os.makedirs(save_dir, exist_ok=True)
    
    # 后处理图像用于保存
    hr_processed = postprocess(hr_images, cfg)
    occluded_hr_processed = postprocess(occluded_hr_images, cfg)
    lr_processed = postprocess(lr_images, cfg)
    restored_processed = postprocess(restored_images, cfg)
    bicubic_processed = postprocess(bicubic_images, cfg)
    
    # 确保值域在[0,1]并转换为uint8
    hr_processed = torch.clamp(hr_processed, 0, 1)
    occluded_hr_processed = torch.clamp(occluded_hr_processed, 0, 1)
    lr_processed = torch.clamp(lr_processed, 0, 1)
    restored_processed = torch.clamp(restored_processed, 0, 1)
    bicubic_processed = torch.clamp(bicubic_processed, 0, 1)
    
    for i in range(len(hr_images)):
        # HR真实图像
        hr_img = (hr_processed[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(hr_img).save(f"{save_dir}/01_HR_GT_{i:02d}.png")
        
        # 遮挡后的HR图像
        occluded_hr_img = (occluded_hr_processed[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(occluded_hr_img).save(f"{save_dir}/02_HR_Occluded_{i:02d}.png")
        
        # LR输入图像
        lr_img = (lr_processed[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(lr_img).save(f"{save_dir}/03_LR_Input_{i:02d}.png")
        
        # 双三次插值结果
        bicubic_img = (bicubic_processed[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(bicubic_img).save(f"{save_dir}/04_Bicubic_Upsampled_{i:02d}.png")
        
        # PnP-Flow恢复结果
        restored_img = (restored_processed[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(restored_img).save(f"{save_dir}/05_Restored_PnPFlow_{i:02d}.png")
        
        print(f"已保存第 {i+1} 组遮挡对比图像")
    
    print("文件命名规则:")
    print("  01_HR_GT_XX.png: 高分辨率真实图像 (Ground Truth)")
    print("  02_HR_Occluded_XX.png: 遮挡后的高分辨率图像")
    print("  03_LR_Input_XX.png: 低分辨率输入图像")
    print("  04_Bicubic_Upsampled_XX.png: 双三次插值上采样结果")
    print("  05_Restored_PnPFlow_XX.png: PnP-Flow恢复的图像")

def calculate_occlusion_metrics(hr_images, restored_images, occluded_hr_images, cfg):
    """计算遮挡恢复的评估指标"""
    print("=== 计算遮挡恢复评估指标 ===")
    
    # 后处理到[0,1]范围用于指标计算
    hr_processed = postprocess(hr_images, cfg)
    restored_processed = postprocess(restored_images, cfg)
    occluded_hr_processed = postprocess(occluded_hr_images, cfg)
    
    # 确保值域在[0,1]
    hr_processed = torch.clamp(hr_processed, 0, 1)
    restored_processed = torch.clamp(restored_processed, 0, 1)
    occluded_hr_processed = torch.clamp(occluded_hr_processed, 0, 1)
    
    # 计算恢复后vs真实HR的PSNR/SSIM
    mse = torch.mean((hr_processed - restored_processed) ** 2)
    psnr_restored = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    # 计算遮挡HR vs 真实HR的PSNR/SSIM（作为基线参考）
    mse_occluded = torch.mean((hr_processed - occluded_hr_processed) ** 2)
    psnr_occluded = 20 * torch.log10(1.0 / torch.sqrt(mse_occluded))
    
    print(f"遮挡图像 vs 真实图像 PSNR: {psnr_occluded.item():.4f} dB")
    print(f"恢复图像 vs 真实图像 PSNR: {psnr_restored.item():.4f} dB")
    
    # 计算SSIM
    def ssim_single_channel(img1, img2):
        mu1 = F.avg_pool2d(img1, 3, 1, 1)
        mu2 = F.avg_pool2d(img2, 3, 1, 1)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    
    # 对每个通道计算SSIM
    ssim_restored = []
    ssim_occluded = []
    for c in range(hr_processed.shape[1]):
        ssim_r = ssim_single_channel(hr_processed[:, c:c+1], restored_processed[:, c:c+1])
        ssim_o = ssim_single_channel(hr_processed[:, c:c+1], occluded_hr_processed[:, c:c+1])
        ssim_restored.append(ssim_r)
        ssim_occluded.append(ssim_o)
    
    ssim_restored = torch.mean(torch.stack(ssim_restored))
    ssim_occluded = torch.mean(torch.stack(ssim_occluded))
    
    print(f"遮挡图像 vs 真实图像 SSIM: {ssim_occluded.item():.4f}")
    print(f"恢复图像 vs 真实图像 SSIM: {ssim_restored.item():.4f}")
    
    return (psnr_occluded.item(), ssim_occluded.item(), 
            psnr_restored.item(), ssim_restored.item())

def calculate_baseline_metrics(hr_images, lr_images, cfg):
    """计算LR双三次插值上采样的基线指标"""
    print("=== 计算双三次插值基线指标 ===")
    
    # 对LR图像进行双三次插值上采样
    sf = hr_images.shape[-1] // lr_images.shape[-1]  # 计算放大倍数
    bicubic_upsampled = F.interpolate(lr_images, scale_factor=sf, mode='bicubic', align_corners=False)
    
    # 后处理到[0,1]范围
    hr_processed = postprocess(hr_images, cfg)
    bicubic_processed = postprocess(bicubic_upsampled, cfg)
    
    # 确保值域在[0,1]
    hr_processed = torch.clamp(hr_processed, 0, 1)
    bicubic_processed = torch.clamp(bicubic_processed, 0, 1)
    
    # 计算PSNR
    mse = torch.mean((hr_processed - bicubic_processed) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    print(f"双三次插值 PSNR: {psnr.item():.4f} dB")
    
    # 计算SSIM
    def ssim_single_channel(img1, img2):
        mu1 = F.avg_pool2d(img1, 3, 1, 1)
        mu2 = F.avg_pool2d(img2, 3, 1, 1)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    
    # 对每个通道计算SSIM
    ssim_values = []
    for c in range(hr_processed.shape[1]):
        ssim_c = ssim_single_channel(hr_processed[:, c:c+1], bicubic_processed[:, c:c+1])
        ssim_values.append(ssim_c)
    
    ssim = torch.mean(torch.stack(ssim_values))
    print(f"双三次插值 SSIM: {ssim.item():.4f}")
    
    return psnr.item(), ssim.item(), bicubic_upsampled

def run_pnp_flow_occlusion_test(model, cfg, device, num_samples=8, save_dir="./test_results_occlusion", 
                                occlusion_params=None):
    """运行PnP-Flow遮挡超分辨率测试"""
    print("=== 开始PnP-Flow遮挡超分辨率测试 ===")
    
    # 创建参数对象
    args = Args(cfg)
    args.save_path = save_dir
    # 调整关键超参数以增强先验作用
    args.steps_pnp = 100
    args.num_samples = 3
    args.lr_pnp = 1.0
    
    # 按照main.py中的设置确定超分辨率倍数
    if cfg.dim_image == 128:
        sf = 2
        print('Superresolution with scale factor 2')
    elif cfg.dim_image == 256:
        sf = 4
        print('Superresolution with scale factor 4')
    elif cfg.dim_image == 512:
        sf = 2  # 对于512维图像，我们测试2x超分辨率
        print('Superresolution with scale factor 2 (512->256->512)')
    else:
        sf = 2  # 默认2x
        print(f'Superresolution with scale factor 2 (default for dim_image={cfg.dim_image})')
    
    # 设置遮挡参数
    if occlusion_params is None:
        occlusion_params = {
            'num_blocks': 5,      # 遮挡方块数量
            'min_size': 30,       # 方块最小尺寸
            'max_size': 80,       # 方块最大尺寸
            'intensity': 0.0      # 遮挡强度 (黑色遮挡)
        }
    
    print(f"遮挡参数: {occlusion_params}")
    print(f"噪声强度: {args.sigma_noise} (gaussian)")
    
    # 创建带遮挡的降质模型
    degradation = OccludedSuperresolution(sf, cfg.dim_image, device, occlusion_params)
    sigma_noise = args.sigma_noise
    
    # 从数据集加载测试数据
    print("从数据集加载测试数据...")
    data_loaders = DataLoaders(cfg.dataset, num_samples, num_samples).load_data()
    test_loader = data_loaders['test']
    
    # 获取HR图像
    hr_images = []
    for batch_idx, (hr_batch, _) in enumerate(test_loader):
        hr_images.append(hr_batch)
        if len(hr_images) * hr_batch.size(0) >= num_samples:
            break
    
    hr_images = torch.cat(hr_images, dim=0)[:num_samples]
    print(f"加载了 {len(hr_images)} 张HR图像，形状: {hr_images.shape}")
    
    # 创建带遮挡的LR图像
    lr_images, occluded_hr_images, lr_masks = create_occluded_lr_images(hr_images, degradation, sigma_noise, device)
    
    # 创建PnP-Flow求解器
    pnp_solver = PNP_FLOW(model, device, args)
    
    # 创建测试数据加载器（用于PnP求解）
    test_dataset = torch.utils.data.TensorDataset(hr_images, torch.zeros(len(hr_images)))
    test_loader_pnp = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print("开始PnP-Flow求解...")
    
    # 临时保存原始的solve_ip方法并修改它
    original_solve_ip = pnp_solver.solve_ip
    
    restored_images = []
    
    def custom_solve_ip(test_loader, degradation, sigma_noise, H_funcs=None):
        H = degradation.H
        H_adj = degradation.H_adj
        args.sigma_noise = sigma_noise
        num_samples_pnp = args.num_samples
        steps, delta = args.steps_pnp, 1 / args.steps_pnp
        
        # 按照main.py中的学习率设置
        if args.noise_type == 'gaussian':
            lr = sigma_noise**2 * args.lr_pnp
            print(f"使用学习率: {lr} (基础lr: {args.lr_pnp}, sigma_noise: {sigma_noise})")
        else:
            raise ValueError('Noise type not supported')
        
        loader = iter(test_loader)
        for batch in range(min(args.max_batch, len(hr_images))):
            (clean_img, _) = next(loader)
            
            noisy_img = lr_images[batch:batch+1].to(device)
            mask = lr_masks[batch:batch+1].to(device)
            clean_img = clean_img.to('cpu')
            
            print(f"\n处理第 {batch+1}/{min(args.max_batch, len(hr_images))} 张图像")
            print(f"LR输入形状: {noisy_img.shape}, 值域: [{noisy_img.min():.3f}, {noisy_img.max():.3f}]")
            print(f"算法参数: steps={steps}, lr={lr}, num_samples={num_samples_pnp}")
            
            # 初始化：使用标准的H_adj
            x = H_adj(noisy_img).to(device)
            print(f"初始化形状: {x.shape}, 值域: [{x.min():.3f}, {x.max():.3f}]")
            
            # PnP-Flow迭代
            with torch.no_grad():
                for iteration in range(int(steps)):
                    t1 = torch.ones(len(x), device=device) * delta * iteration
                    lr_t = pnp_solver.learning_rate_strat(lr, t1)
                    
                    # 仅在非遮挡像素计算梯度
                    residual = H(x) - noisy_img
                    residual = residual * mask  # 忽略遮挡像素
                    z = x - lr_t * H_adj(residual) / (sigma_noise**2)
                    
                    # 先验项（使用Flow Matching模型）
                    x_new = torch.zeros_like(x)
                    for _ in range(num_samples_pnp):
                        z_tilde = pnp_solver.interpolation_step(z, t1.view(-1, 1, 1, 1))
                        x_new += pnp_solver.denoiser(z_tilde, t1)
                    
                    x_new /= num_samples_pnp
                    x = x_new
                    
                    # 打印进度
                    print_interval = max(1, steps // 5)  # 打印5次进度
                    if (iteration + 1) % print_interval == 0 or iteration == 0:
                        print(f"  迭代 {iteration+1}/{int(steps)}, 值域: [{x.min():.3f}, {x.max():.3f}]")
            
            restored_img = x.detach().clone()
            restored_images.append(restored_img.cpu())
            
            print(f"恢复完成，最终值域: [{restored_img.min():.3f}, {restored_img.max():.3f}]")
    
    # 替换solve_ip方法
    pnp_solver.solve_ip = custom_solve_ip
    
    # 运行求解
    pnp_solver.solve_ip(test_loader_pnp, degradation, sigma_noise)
    
    # 恢复原始方法
    pnp_solver.solve_ip = original_solve_ip
    
    # 合并恢复的图像
    restored_images = torch.cat(restored_images, dim=0)
    
    print(f"\n=== PnP-Flow遮挡恢复完成 ===")
    print(f"处理了 {len(restored_images)} 张图像")
    
    return hr_images, occluded_hr_images, lr_images, restored_images

def main():
    parser = argparse.ArgumentParser(description='PnP-Flow带遮挡超分辨率测试')
    parser.add_argument('--model_path', type=str, default='./model/cropsr/ot/model_85.pt',
                       help='模型路径')
    parser.add_argument('--num_samples', type=int, default=8,
                       help='测试样本数量')
    parser.add_argument('--save_images', action='store_true', default=True,
                       help='保存样本图像')
    parser.add_argument('--num_blocks', type=int, default=5,
                       help='遮挡方块数量')
    parser.add_argument('--min_size', type=int, default=30,
                       help='遮挡方块最小尺寸')
    parser.add_argument('--max_size', type=int, default=80,
                       help='遮挡方块最大尺寸')
    parser.add_argument('--intensity', type=float, default=0.0,
                       help='遮挡强度 (0.0=黑色, 1.0=白色, 0.5=灰色)')
    args = parser.parse_args()
    
    try:
        # 检查GPU
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        if torch.cuda.is_available():
            print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # 清理显存
        clear_gpu_memory()
        
        # 加载模型
        model, cfg = load_model_and_config(args.model_path, device)
        
        # 设置遮挡参数
        occlusion_params = {
            'num_blocks': args.num_blocks,
            'min_size': args.min_size,
            'max_size': args.max_size,
            'intensity': args.intensity
        }
        
        # 运行PnP-Flow遮挡测试
        hr_images, occluded_hr_images, lr_images, restored_images = run_pnp_flow_occlusion_test(
            model, cfg, device, 
            num_samples=args.num_samples,
            save_dir="./test_results_occlusion",
            occlusion_params=occlusion_params
        )
        
        # 计算双三次插值基线指标
        baseline_psnr, baseline_ssim, bicubic_images = calculate_baseline_metrics(hr_images, lr_images, cfg)
        
        # 计算遮挡恢复指标
        psnr_occluded, ssim_occluded, psnr_restored, ssim_restored = calculate_occlusion_metrics(
            hr_images, restored_images, occluded_hr_images, cfg)
        
        # 保存对比图像
        if args.save_images:
            save_occlusion_comparison_images(hr_images, occluded_hr_images, lr_images, restored_images, bicubic_images, cfg)
        
        # 详细对比结果
        print("\n" + "="*70)
        print("农业航拍图像带遮挡2倍超分辨率效果对比")
        print("="*70)
        print(f"测试样本数量: {len(hr_images)}")
        print(f"超分辨率倍数: 2x ({lr_images.shape[-2:][0]}×{lr_images.shape[-2:][1]} → {hr_images.shape[-2:][0]}×{hr_images.shape[-2:][1]})")
        print(f"数据集: CropSR农业航拍图像")
        print(f"遮挡参数: {occlusion_params}")
        print("-" * 70)
        
        print("📊 指标对比:")
        print(f"{'方法':<25} {'PSNR (dB)':<12} {'SSIM':<8} {'说明'}")
        print("-" * 70)
        print(f"{'遮挡图像 (损失基线)':<25} {psnr_occluded:<12.4f} {ssim_occluded:<8.4f} 遮挡造成的损失")
        print(f"{'双三次插值':<25} {baseline_psnr:<12.4f} {baseline_ssim:<8.4f} 传统插值方法")
        print(f"{'PnP-Flow (我们的)':<25} {psnr_restored:<12.4f} {ssim_restored:<8.4f} Flow Matching恢复")
        print(f"{'HR真实图像':<25} {'∞':<12} {'1.0000':<8} 理想上限")
        
        print("-" * 70)
        print("📈 遮挡恢复能力:")
        psnr_recovery = psnr_restored - psnr_occluded
        ssim_recovery = ssim_restored - ssim_occluded
        print(f"PSNR 恢复: {psnr_recovery:+.4f} dB ({psnr_recovery/abs(psnr_occluded)*100:+.2f}%)")
        print(f"SSIM 恢复: {ssim_recovery:+.4f} ({ssim_recovery/ssim_occluded*100:+.2f}%)")
        
        print("\n📈 相对于双三次插值的改进:")
        psnr_improvement = psnr_restored - baseline_psnr
        ssim_improvement = ssim_restored - baseline_ssim
        print(f"PSNR 提升: {psnr_improvement:+.4f} dB ({psnr_improvement/baseline_psnr*100:+.2f}%)")
        print(f"SSIM 提升: {ssim_improvement:+.4f} ({ssim_improvement/baseline_ssim*100:+.2f}%)")
        
        print("-" * 70)
        print("🎯 结论:")
        if psnr_recovery > 0 and ssim_recovery > 0:
            print("✅ PnP-Flow成功恢复了遮挡区域的信息")
            print("✅ 证明了Flow Matching先验在遮挡恢复任务中的有效性")
            print("✅ 恢复后的图像质量显著优于遮挡图像")
        else:
            print("⚠️  遮挡恢复效果有限，可能需要调整算法参数")
            
        if psnr_improvement > 0 and ssim_improvement > 0:
            print("✅ PnP-Flow在遮挡+超分辨率组合任务上优于传统方法")
        else:
            print("⚠️  在组合任务上相对传统方法的优势有限")
            
        print(f"\n📁 对比图像已保存到 ./test_results_occlusion/ 目录")
        print("   可以直观比较五种图像的视觉效果：")
        print("   1. 原始高分辨率图像")
        print("   2. 遮挡后的图像")
        print("   3. 降采样的低分辨率图像")
        print("   4. 双三次插值恢复")
        print("   5. PnP-Flow恢复")
        print("="*70)
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
