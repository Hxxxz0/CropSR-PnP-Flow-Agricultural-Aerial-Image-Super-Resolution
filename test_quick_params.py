#!/usr/bin/env python3
"""
快速参数测试脚本 - 可自定义参数范围
只在一张图片上测试以快速评估效果
"""

import torch
import numpy as np
import sys
import os
import argparse
from PIL import Image
import torch.nn.functional as F
import random
from itertools import product

# 添加项目路径
sys.path.append('.')
from pnpflow.utils import define_model, load_cfg_from_cfg_file, postprocess
from pnpflow.dataloaders import DataLoaders
from pnpflow.methods.pnp_flow import PNP_FLOW
from test_model_evaluation_with_occlusion import OccludedSuperresolution, Args, load_model_and_config, create_occluded_lr_images

def run_single_test(model, cfg, device, lr_pnp, steps_pnp, hr_image, degradation, sigma_noise, lr_image, lr_mask):
    """运行单次测试"""
    print(f"测试: lr={lr_pnp}, steps={steps_pnp}")
    
    # 创建参数对象
    args = Args(cfg)
    args.steps_pnp = steps_pnp
    args.lr_pnp = lr_pnp
    args.num_samples = 3
    args.gamma_style = 'alpha_1_minus_t'
    args.alpha = 1.0
    args.max_batch = 1
    
    # 创建PnP-Flow求解器
    pnp_solver = PNP_FLOW(model, device, args)
    
    # 准备数据
    hr_single = hr_image.unsqueeze(0)
    lr_single = lr_image.unsqueeze(0).to(device)
    mask_single = lr_mask.unsqueeze(0).to(device)
    
    test_dataset = torch.utils.data.TensorDataset(hr_single, torch.zeros(1))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    restored_image = None
    
    def custom_solve_ip(test_loader, degradation, sigma_noise, H_funcs=None):
        nonlocal restored_image
        H = degradation.H
        H_adj = degradation.H_adj
        args.sigma_noise = sigma_noise
        num_samples_pnp = args.num_samples
        steps, delta = args.steps_pnp, 1 / args.steps_pnp
        
        # 学习率计算
        if args.noise_type == 'gaussian':
            lr = sigma_noise**2 * args.lr_pnp
        else:
            raise ValueError('Noise type not supported')
        
        loader = iter(test_loader)
        (clean_img, _) = next(loader)
        
        noisy_img = lr_single
        mask = mask_single
        
        # 初始化
        x = H_adj(noisy_img).to(device)
        
        # PnP-Flow迭代
        with torch.no_grad():
            for iteration in range(int(steps)):
                t1 = torch.ones(len(x), device=device) * delta * iteration
                lr_t = pnp_solver.learning_rate_strat(lr, t1)
                
                # 数据保真项
                residual = H(x) - noisy_img
                residual = residual * mask
                z = x - lr_t * H_adj(residual) / (sigma_noise**2)
                
                # 先验项
                x_new = torch.zeros_like(x)
                for _ in range(num_samples_pnp):
                    z_tilde = pnp_solver.interpolation_step(z, t1.view(-1, 1, 1, 1))
                    x_new += pnp_solver.denoiser(z_tilde, t1)
                
                x_new /= num_samples_pnp
                x = x_new
        
        restored_image = x.detach().clone().cpu()
    
    # 运行测试
    original_solve_ip = pnp_solver.solve_ip
    pnp_solver.solve_ip = custom_solve_ip
    pnp_solver.solve_ip(test_loader, degradation, sigma_noise)
    pnp_solver.solve_ip = original_solve_ip
    
    # 计算PSNR
    if restored_image is not None:
        hr_processed = postprocess(hr_single, cfg)
        restored_processed = postprocess(restored_image, cfg)
        
        hr_processed = torch.clamp(hr_processed, 0, 1)
        restored_processed = torch.clamp(restored_processed, 0, 1)
        
        mse = torch.mean((hr_processed - restored_processed) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        return restored_image, psnr.item()
    
    return None, 0.0

def main():
    parser = argparse.ArgumentParser(description='快速参数测试')
    parser.add_argument('--model_path', type=str, default='./model/cropsr/ot/model_85.pt')
    parser.add_argument('--save_dir', type=str, default='./quick_param_test')
    parser.add_argument('--lr_range', nargs='+', type=float, default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                       help='学习率范围 (例如: 0.5 1.0 1.5)')
    parser.add_argument('--steps_range', nargs='+', type=int, default=[50, 100, 200, 300, 400, 500],
                       help='迭代步数范围 (例如: 50 100 200)')
    args = parser.parse_args()
    
    try:
        # 设备和模型
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model, cfg = load_model_and_config(args.model_path, device)
        
        # 创建保存目录
        os.makedirs(args.save_dir, exist_ok=True)
        
        # 设置
        occlusion_params = {
            'num_blocks': 5,
            'min_size': 30,
            'max_size': 80,
            'intensity': 0.0
        }
        
        sf = 2 if cfg.dim_image == 128 else (4 if cfg.dim_image == 256 else 2)
        degradation = OccludedSuperresolution(sf, cfg.dim_image, device, occlusion_params)
        sigma_noise = 0.05
        
        # 加载测试图片
        data_loaders = DataLoaders(cfg.dataset, 1, 1).load_data()
        test_loader = data_loaders['test']
        
        for hr_batch, _ in test_loader:
            hr_image = hr_batch[0]
            break
        
        # 创建遮挡图像
        hr_images_batch = hr_image.unsqueeze(0)
        lr_images, occluded_hr_images, lr_masks = create_occluded_lr_images(
            hr_images_batch, degradation, sigma_noise, device, seed=42)
        
        lr_image = lr_images[0]
        lr_mask = lr_masks[0]
        
        # 保存参考图像
        def save_image(tensor, filename):
            processed = postprocess(tensor, cfg)
            processed = torch.clamp(processed, 0, 1)
            img = (processed[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(args.save_dir, filename))
        
        save_image(hr_images_batch, "00_original_HR.png")
        save_image(occluded_hr_images, "01_occluded_HR.png")
        save_image(lr_images, "02_input_LR.png")
        
        # 双三次插值基线
        bicubic = F.interpolate(lr_images, scale_factor=sf, mode='bicubic', align_corners=False)
        save_image(bicubic, "03_bicubic.png")
        
        print(f"\n=== 开始测试 ===")
        print(f"学习率: {args.lr_range}")
        print(f"步数: {args.steps_range}")
        print(f"总组合数: {len(args.lr_range) * len(args.steps_range)}")
        
        results = []
        
        # 测试所有组合
        for i, (lr_pnp, steps_pnp) in enumerate(product(args.lr_range, args.steps_range)):
            print(f"\n--- {i+1}/{len(args.lr_range) * len(args.steps_range)} ---")
            
            restored_image, psnr = run_single_test(
                model, cfg, device, lr_pnp, steps_pnp, 
                hr_image, degradation, sigma_noise, lr_image, lr_mask
            )
            
            if restored_image is not None:
                # 保存结果
                filename = f"restored_lr{lr_pnp}_steps{steps_pnp}.png"
                save_image(restored_image, filename)
                
                results.append((lr_pnp, steps_pnp, psnr, filename))
                print(f"PSNR: {psnr:.4f} dB -> {filename}")
            
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 结果总结
        print("\n" + "="*60)
        print("测试结果 (按PSNR排序)")
        print("="*60)
        print(f"{'排名':<4} {'学习率':<8} {'步数':<6} {'PSNR':<10} {'文件名'}")
        print("-" * 60)
        
        # 按PSNR排序
        results.sort(key=lambda x: x[2], reverse=True)
        
        for rank, (lr_pnp, steps_pnp, psnr, filename) in enumerate(results, 1):
            print(f"{rank:<4} {lr_pnp:<8} {steps_pnp:<6} {psnr:<10.4f} {filename}")
        
        # 最佳结果
        if results:
            best_lr, best_steps, best_psnr, best_file = results[0]
            print(f"\n🏆 最佳组合:")
            print(f"   学习率: {best_lr}")
            print(f"   步数: {best_steps}")
            print(f"   PSNR: {best_psnr:.4f} dB")
            print(f"   文件: {best_file}")
        
        print(f"\n📁 结果保存在: {args.save_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 