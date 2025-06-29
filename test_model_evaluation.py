#!/usr/bin/env python3
"""
使用PnP-Flow方法测试CropSR超分辨率任务
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

# 添加项目路径
sys.path.append('.')
from pnpflow.utils import define_model, load_cfg_from_cfg_file, postprocess
from pnpflow.train_flow_matching import FLOW_MATCHING
from pnpflow.dataloaders import DataLoaders
from pnpflow.degradations import Superresolution
from pnpflow.methods.pnp_flow import PNP_FLOW
import pnpflow.utils as utils

class SimpleSuperresolution:
    """简化的超分辨率降质类，避免创建巨大的降采样矩阵"""
    def __init__(self, sf, dim_image, device="cuda"):
        self.sf = sf
        self.device = device
        
    def H(self, x):
        """降采样操作：HR -> LR"""
        return utils.downsample(x, self.sf)
    
    def H_adj(self, x):
        """上采样操作：LR -> HR (双三次插值)"""
        return utils.upsample(x, self.sf)

class Args:
    """配置类，模拟argparse参数"""
    def __init__(self, cfg, num_samples=8):
        # 从配置文件复制基本参数
        for key, value in cfg.items():
            setattr(self, key, value)
        
        # PnP-Flow特定参数
        self.method = 'pnp_flow'
        self.problem = 'superresolution'
        self.noise_type = 'gaussian'
        self.sigma_noise = 0.05
        self.max_batch = num_samples  # 使用传入的样本数量
        self.batch_size_ip = 1  # 每次处理一张图像
        self.eval_split = 'test'
        
        # PnP-Flow算法参数 4x
        self.steps_pnp = 200  # PnP迭代步数
        self.lr_pnp = 1.5     # 学习率
        self.num_samples = 3   # 每步采样数量（增加采样提高稳定性）
        self.gamma_style = 'constant'  # 学习率策略
        self.alpha = 1.0

        # PnP-Flow算法参数 8x
        # self.steps_pnp = 150  # PnP迭代步数
        # self.lr_pnp = 2.0     # 学习率
        # self.num_samples = 8   # 每步采样数量（增加采样提高稳定性）
        # self.gamma_style = 'constant'  # 学习率策略
        # self.alpha = 1.0
        
        # 保存和计算选项
        self.save_results = True
        self.compute_time = False
        self.compute_memory = False
        
        # 创建字典用于保存路径
        self.dict_cfg_method = {
            'steps_pnp': self.steps_pnp,
            'lr_pnp': self.lr_pnp,
            'num_samples': self.num_samples,
            'gamma_style': self.gamma_style
        }

def load_8x_data(lr_path, hr_path, num_samples, device, start_idx=0):
    """从指定路径加载8倍超分辨率数据"""
    print(f"=== 加载8倍超分辨率数据 ===")
    print(f"LR路径: {lr_path}")
    print(f"HR路径: {hr_path}")
    print(f"起始索引: {start_idx}")
    
    # 获取文件列表
    lr_files = sorted([f for f in os.listdir(lr_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    hr_files = sorted([f for f in os.listdir(hr_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"找到 {len(lr_files)} 个LR文件, {len(hr_files)} 个HR文件")
    
    # 确保起始索引有效
    max_start_idx = min(len(lr_files), len(hr_files)) - num_samples
    if start_idx > max_start_idx:
        start_idx = max_start_idx
        print(f"调整起始索引为: {start_idx}")
    
    # 选择指定范围的文件
    end_idx = start_idx + num_samples
    selected_lr_files = lr_files[start_idx:end_idx]
    selected_hr_files = hr_files[start_idx:end_idx]
    
    print(f"选择文件范围: {start_idx} 到 {end_idx-1}")
    print(f"第一个文件: {selected_lr_files[0]} / {selected_hr_files[0]}")
    print(f"最后一个文件: {selected_lr_files[-1]} / {selected_hr_files[-1]}")
    
    lr_images = []
    hr_images = []
    
    for i, (lr_file, hr_file) in enumerate(zip(selected_lr_files, selected_hr_files)):
        # 加载LR图像
        lr_img = Image.open(os.path.join(lr_path, lr_file)).convert('RGB')
        lr_tensor = torch.from_numpy(np.array(lr_img)).float().permute(2, 0, 1) / 255.0
        lr_images.append(lr_tensor)
        
        # 加载HR图像
        hr_img = Image.open(os.path.join(hr_path, hr_file)).convert('RGB')
        hr_tensor = torch.from_numpy(np.array(hr_img)).float().permute(2, 0, 1) / 255.0
        hr_images.append(hr_tensor)
        
        if i == 0:
            print(f"LR图像尺寸: {lr_tensor.shape}")
            print(f"HR图像尺寸: {hr_tensor.shape}")
    
    lr_images = torch.stack(lr_images)
    hr_images = torch.stack(hr_images)
    
    print(f"加载完成: LR {lr_images.shape}, HR {hr_images.shape}")
    return lr_images, hr_images

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

def create_lr_images(hr_images, degradation, sigma_noise, device, seed=42):
    """创建LR图像（模拟真实的超分辨率任务）"""
    print("=== 创建LR图像 ===")
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(seed)
    
    lr_images = []
    for i, hr_img in enumerate(hr_images):
        hr_single = hr_img.unsqueeze(0).to(device)
        
        # 应用降质操作
        lr_single = degradation.H(hr_single)
        
        # 添加噪声
        if sigma_noise > 0:
            lr_single += torch.randn_like(lr_single) * sigma_noise
        
        lr_images.append(lr_single.cpu())
        
        print(f"创建第 {i+1} 张LR图像")
        print(f"  HR形状: {hr_single.shape}, 值域: [{hr_single.min():.3f}, {hr_single.max():.3f}]")
        print(f"  LR形状: {lr_single.shape}, 值域: [{lr_single.min():.3f}, {lr_single.max():.3f}]")
    
    return torch.cat(lr_images, dim=0)

def save_comparison_images(hr_images, lr_images, restored_images, bicubic_images, cfg, save_dir="./test_results"):
    """保存四种图像的对比：HR真实、LR输入、双三次插值、PnP-Flow恢复"""
    print(f"=== 保存对比图像到 {save_dir} ===")
    os.makedirs(save_dir, exist_ok=True)
    
    # 后处理图像用于保存
    hr_processed = postprocess(hr_images, cfg)
    lr_processed = postprocess(lr_images, cfg)
    restored_processed = postprocess(restored_images, cfg)
    bicubic_processed = postprocess(bicubic_images, cfg)
    
    # 确保值域在[0,1]并转换为uint8
    hr_processed = torch.clamp(hr_processed, 0, 1)
    lr_processed = torch.clamp(lr_processed, 0, 1)
    restored_processed = torch.clamp(restored_processed, 0, 1)
    bicubic_processed = torch.clamp(bicubic_processed, 0, 1)
    
    for i in range(len(hr_images)):
        # HR真实图像
        hr_img = (hr_processed[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(hr_img).save(f"{save_dir}/HR_GT_{i:02d}.png")
        
        # LR输入图像
        lr_img = (lr_processed[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(lr_img).save(f"{save_dir}/LR_Input_{i:02d}.png")
        
        # 双三次插值结果
        bicubic_img = (bicubic_processed[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(bicubic_img).save(f"{save_dir}/Bicubic_Upsampled_{i:02d}.png")
        
        # PnP-Flow恢复结果
        restored_img = (restored_processed[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(restored_img).save(f"{save_dir}/Restored_PnPFlow_{i:02d}.png")
        
        print(f"已保存第 {i+1} 组对比图像")
    
    print("文件命名规则:")
    print("  HR_GT_XX.png: 高分辨率真实图像 (Ground Truth)")
    print("  LR_Input_XX.png: 低分辨率输入图像")
    print("  Bicubic_Upsampled_XX.png: 双三次插值上采样结果")
    print("  Restored_PnPFlow_XX.png: PnP-Flow恢复的图像")

def calculate_metrics(hr_images, restored_images, cfg):
    """计算PSNR和SSIM指标"""
    print("=== 计算评估指标 ===")
    
    # 后处理到[0,1]范围用于指标计算
    hr_processed = postprocess(hr_images, cfg)
    restored_processed = postprocess(restored_images, cfg)
    
    # 确保值域在[0,1]
    hr_processed = torch.clamp(hr_processed, 0, 1)
    restored_processed = torch.clamp(restored_processed, 0, 1)
    
    # 计算PSNR
    mse = torch.mean((hr_processed - restored_processed) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    print(f"PSNR: {psnr.item():.4f} dB")
    
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
        ssim_c = ssim_single_channel(hr_processed[:, c:c+1], restored_processed[:, c:c+1])
        ssim_values.append(ssim_c)
    
    ssim = torch.mean(torch.stack(ssim_values))
    print(f"SSIM: {ssim.item():.4f}")
    
    return psnr.item(), ssim.item()

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

def run_pnp_flow_test(model, cfg, device, num_samples=8, save_dir="./test_results", 
                      scale_factor=2, lr_data_path=None, hr_data_path=None, start_idx=0):
    """运行PnP-Flow超分辨率测试"""
    print("=== 开始PnP-Flow超分辨率测试 ===")
    
    # 创建参数对象
    args = Args(cfg, num_samples)
    args.save_path = save_dir
    
    # 设置超分辨率倍数
    sf = scale_factor
    print(f'超分辨率倍数: {sf}x')
    
    # 根据是否提供外部数据路径选择数据加载方式
    if scale_factor == 8 and lr_data_path and hr_data_path:
        # 8倍超分辨率：从指定路径加载数据
        lr_images, hr_images = load_8x_data(lr_data_path, hr_data_path, num_samples, device, start_idx)
        print(f"从外部路径加载了 {len(hr_images)} 张图像")
        print(f"LR形状: {lr_images.shape}, HR形状: {hr_images.shape}")
        
        # 验证尺寸关系
        actual_sf = hr_images.shape[-1] // lr_images.shape[-1]
        print(f"实际超分辨率倍数: {actual_sf}x")
        if actual_sf != sf:
            print(f"警告: 期望倍数 {sf}x，但实际倍数为 {actual_sf}x")
            sf = actual_sf  # 使用实际倍数
        
        # 使用HR图像的尺寸作为目标尺寸
        target_dim = hr_images.shape[-1]
        degradation = SimpleSuperresolution(sf, target_dim, device)
        sigma_noise = args.sigma_noise
        
        print(f"使用外部数据，目标尺寸: {target_dim}")
    else:
        # 2x/4x超分辨率：从数据集加载并生成LR图像
        print("从数据集加载测试数据...")
        
        # 使用bicubic模式避免巨大的降采样矩阵
        print("使用双三次插值模式进行超分辨率（避免显存问题）")
        degradation = SimpleSuperresolution(sf, cfg.dim_image, device)
        sigma_noise = args.sigma_noise
        
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
        
        # 创建LR图像
        lr_images = create_lr_images(hr_images, degradation, sigma_noise, device)
    
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
        
        # 传递scale_factor到内部函数
        current_sf = scale_factor
        
        if args.noise_type == 'gaussian':
            args.lr_pnp = sigma_noise**2 * args.lr_pnp
            lr = args.lr_pnp
        else:
            raise ValueError('Noise type not supported')
        
        loader = iter(test_loader)
        for batch in range(min(args.max_batch, len(hr_images))):
            (clean_img, _) = next(loader)
            
            # 使用预先创建的LR图像
            noisy_img = lr_images[batch:batch+1].to(device)
            clean_img = clean_img.to('cpu')
            
            print(f"\n处理第 {batch+1}/{min(args.max_batch, len(hr_images))} 张图像")
            print(f"LR输入形状: {noisy_img.shape}, 值域: [{noisy_img.min():.3f}, {noisy_img.max():.3f}]")
            
            # 初始化策略选择
            if current_sf >= 4:
                # 对于4倍及以上超分辨率：先用双三次插值，再用PnP-Flow优化
                print("使用两阶段策略：双三次插值 + PnP-Flow优化")
                x = F.interpolate(noisy_img, scale_factor=current_sf, mode='bicubic', align_corners=False).to(device)
                print(f"双三次插值初始化形状: {x.shape}, 值域: [{x.min():.3f}, {x.max():.3f}]")
            else:
                # 对于2x超分辨率：使用原始的H_adj初始化
                x = H_adj(noisy_img).to(device)
                print(f"H_adj初始化形状: {x.shape}, 值域: [{x.min():.3f}, {x.max():.3f}]")
            
            # PnP-Flow迭代
            with torch.no_grad():
                for iteration in range(int(steps)):
                    t1 = torch.ones(len(x), device=device) * delta * iteration
                    lr_t = pnp_solver.learning_rate_strat(lr, t1)
                    
                    # 数据保真项梯度
                    z = x - lr_t * pnp_solver.grad_datafit(x, noisy_img, H, H_adj)
                    
                    # 先验项（使用Flow Matching模型）
                    x_new = torch.zeros_like(x)
                    for _ in range(num_samples_pnp):
                        z_tilde = pnp_solver.interpolation_step(z, t1.view(-1, 1, 1, 1))
                        x_new += pnp_solver.denoiser(z_tilde, t1)
                    
                    x_new /= num_samples_pnp
                    x = x_new
                    
                    if (iteration + 1) % 20 == 0:
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
    
    print(f"\n=== PnP-Flow求解完成 ===")
    print(f"处理了 {len(restored_images)} 张图像")
    
    return hr_images, lr_images, restored_images

def main():
    parser = argparse.ArgumentParser(description='PnP-Flow超分辨率测试')
    parser.add_argument('--model_path', type=str, default='./model/cropsr/ot/model_85.pt',
                       help='模型路径')
    parser.add_argument('--num_samples', type=int, default=4,
                       help='测试样本数量')
    parser.add_argument('--save_images', action='store_true', default=True,
                       help='保存样本图像')
    parser.add_argument('--scale_factor', type=int, choices=[2, 4, 8], default=2,
                       help='超分辨率倍数 (2x, 4x, 8x)')
    parser.add_argument('--lr_data_path', type=str, default=None,
                       help='低分辨率数据路径 (用于8x超分辨率)')
    parser.add_argument('--hr_data_path', type=str, default=None,
                       help='高分辨率数据路径 (用于8x超分辨率)')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='数据加载起始索引 (用于选择不同批次的数据)')
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
        
        # 运行PnP-Flow测试
        hr_images, lr_images, restored_images = run_pnp_flow_test(
            model, cfg, device, 
            num_samples=args.num_samples,
            save_dir="./test_results",
            scale_factor=args.scale_factor,
            lr_data_path=args.lr_data_path,
            hr_data_path=args.hr_data_path,
            start_idx=args.start_idx
        )
        
        # 确保用于评估的图像数量一致
        num_restored = len(restored_images)
        hr_images = hr_images[:num_restored]
        lr_images = lr_images[:num_restored]
        
        # 计算双三次插值基线指标
        baseline_psnr, baseline_ssim, bicubic_images = calculate_baseline_metrics(hr_images, lr_images, cfg)
        
        # 计算PnP-Flow增强后的指标
        pnpflow_psnr, pnpflow_ssim = calculate_metrics(hr_images, restored_images, cfg)
        
        # 保存对比图像
        if args.save_images:
            save_comparison_images(hr_images, lr_images, restored_images, bicubic_images, cfg)
        
        # 详细对比结果
        print("\n" + "="*60)
        if args.scale_factor == 8:
            print("农业航拍图像8倍超分辨率效果对比")
        else:
            print("农业航拍图像超分辨率效果对比")
        print("="*60)
        print(f"测试样本数量: {len(hr_images)}")
        actual_sf = hr_images.shape[-1] // lr_images.shape[-1]
        print(f"超分辨率倍数: {actual_sf}x ({lr_images.shape[-2:][0]}×{lr_images.shape[-2:][1]} → {hr_images.shape[-2:][0]}×{hr_images.shape[-2:][1]})")
        if args.scale_factor == 8:
            print(f"数据集: CropSR农业航拍图像 (8倍超分辨率)")
        else:
            print(f"数据集: CropSR农业航拍图像")
        print("-" * 60)
        
        print("📊 指标对比:")
        print(f"{'方法':<20} {'PSNR (dB)':<12} {'SSIM':<8} {'说明'}")
        print("-" * 60)
        print(f"{'双三次插值 (基线)':<20} {baseline_psnr:<12.4f} {baseline_ssim:<8.4f} 传统插值方法")
        print(f"{'PnP-Flow (我们的)':<20} {pnpflow_psnr:<12.4f} {pnpflow_ssim:<8.4f} Flow Matching增强")
        print(f"{'HR真实图像':<20} {'∞':<12} {'1.0000':<8} 理想上限")
        
        print("-" * 60)
        print("📈 改进幅度:")
        psnr_improvement = pnpflow_psnr - baseline_psnr
        ssim_improvement = pnpflow_ssim - baseline_ssim
        print(f"PSNR 提升: {psnr_improvement:+.4f} dB ({psnr_improvement/baseline_psnr*100:+.2f}%)")
        print(f"SSIM 提升: {ssim_improvement:+.4f} ({ssim_improvement/baseline_ssim*100:+.2f}%)")
        
        print("-" * 60)
        print("🎯 结论:")
        if psnr_improvement > 0 and ssim_improvement > 0:
            if args.scale_factor == 8:
                print("✅ PnP-Flow在8倍超分辨率这一极具挑战性的任务上显著优于传统方法")
                print("✅ 证明了Flow Matching先验在极高倍数超分辨率中的有效性")
                print("✅ 为农业航拍图像的精细分析提供了强有力的技术支撑")
            else:
                print("✅ PnP-Flow方法在PSNR和SSIM上都优于双三次插值基线")
                print("✅ 证明了Flow Matching先验在农业图像超分辨率中的有效性")
        elif psnr_improvement > 0:
            print("✅ PnP-Flow在PSNR上优于基线，但SSIM提升有限")
        elif ssim_improvement > 0:
            print("✅ PnP-Flow在SSIM上优于基线，但PSNR提升有限")
        else:
            if args.scale_factor == 8:
                print("⚠️  8倍超分辨率是极具挑战性的任务，需要进一步调优参数")
            else:
                print("⚠️  PnP-Flow方法需要进一步调优")
            
        print(f"\n📁 对比图像已保存到 ./test_results/ 目录")
        print("   可以直观比较四种图像的视觉效果")
        print("="*60)
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 