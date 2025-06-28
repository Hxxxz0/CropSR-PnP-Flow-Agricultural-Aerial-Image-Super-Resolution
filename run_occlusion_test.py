#!/usr/bin/env python3
"""
运行带遮挡的2X超分辨率测试的便捷脚本
演示不同遮挡参数的效果
"""

import os
import subprocess
import sys

def run_test(model_path, num_samples=8, **occlusion_params):
    """运行遮挡测试"""
    cmd = [
        sys.executable, 'test_model_evaluation_with_occlusion.py',
        '--model_path', model_path,
        '--num_samples', str(num_samples),
        '--save_images'
    ]
    
    # 添加遮挡参数
    if 'num_blocks' in occlusion_params:
        cmd.extend(['--num_blocks', str(occlusion_params['num_blocks'])])
    if 'min_size' in occlusion_params:
        cmd.extend(['--min_size', str(occlusion_params['min_size'])])
    if 'max_size' in occlusion_params:
        cmd.extend(['--max_size', str(occlusion_params['max_size'])])
    if 'intensity' in occlusion_params:
        cmd.extend(['--intensity', str(occlusion_params['intensity'])])
    
    print(f"运行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

def main():
    # 使用最新模型路径
    model_path = './model/cropsr/ot/model_15.pt'
    
    print("="*60)
    print("PnP-Flow 带遮挡2X超分辨率测试")
    print("="*60)
    
    # 测试场景1：轻度遮挡
    print("\n🔍 测试场景1：轻度遮挡（3个小方块）")
    success = run_test(
        model_path=model_path,
        num_samples=6,
        num_blocks=3,      # 3个遮挡方块
        min_size=25,       # 最小尺寸25像素
        max_size=50,       # 最大尺寸50像素
        intensity=0.0      # 黑色遮挡
    )
    
    if success:
        print("✅ 轻度遮挡测试完成")
    else:
        print("❌ 轻度遮挡测试失败")
        return
    
    # 重命名结果文件夹
    if os.path.exists('./test_results_occlusion'):
        os.rename('./test_results_occlusion', './test_results_light_occlusion')
        print("📁 结果保存至: ./test_results_light_occlusion/")
    
    print("\n" + "-"*60)
    
    # 测试场景2：中等遮挡
    print("\n🔍 测试场景2：中等遮挡（5个中等方块）")
    success = run_test(
        model_path=model_path,
        num_samples=6,
        num_blocks=5,      # 5个遮挡方块
        min_size=30,       # 最小尺寸30像素
        max_size=80,       # 最大尺寸80像素
        intensity=0.0      # 黑色遮挡
    )
    
    if success:
        print("✅ 中等遮挡测试完成")
    else:
        print("❌ 中等遮挡测试失败")
        return
    
    # 重命名结果文件夹
    if os.path.exists('./test_results_occlusion'):
        os.rename('./test_results_occlusion', './test_results_medium_occlusion')
        print("📁 结果保存至: ./test_results_medium_occlusion/")
    
    print("\n" + "-"*60)
    
    # 测试场景3：重度遮挡
    print("\n🔍 测试场景3：重度遮挡（7个大方块）")
    success = run_test(
        model_path=model_path,
        num_samples=6,
        num_blocks=7,      # 7个遮挡方块
        min_size=40,       # 最小尺寸40像素
        max_size=100,      # 最大尺寸100像素
        intensity=0.0      # 黑色遮挡
    )
    
    if success:
        print("✅ 重度遮挡测试完成")
    else:
        print("❌ 重度遮挡测试失败")
        return
    
    # 重命名结果文件夹
    if os.path.exists('./test_results_occlusion'):
        os.rename('./test_results_occlusion', './test_results_heavy_occlusion')
        print("📁 结果保存至: ./test_results_heavy_occlusion/")
    
    print("\n" + "-"*60)
    
    # 测试场景4：灰色遮挡
    print("\n🔍 测试场景4：灰色遮挡（模拟阴影）")
    success = run_test(
        model_path=model_path,
        num_samples=6,
        num_blocks=4,      # 4个遮挡方块
        min_size=35,       # 最小尺寸35像素
        max_size=70,       # 最大尺寸70像素
        intensity=0.3      # 灰色遮挡（模拟阴影）
    )
    
    if success:
        print("✅ 灰色遮挡测试完成")
    else:
        print("❌ 灰色遮挡测试失败")
        return
    
    # 重命名结果文件夹
    if os.path.exists('./test_results_occlusion'):
        os.rename('./test_results_occlusion', './test_results_gray_occlusion')
        print("📁 结果保存至: ./test_results_gray_occlusion/")
    
    print("\n" + "="*60)
    print("🎉 所有遮挡测试完成！")
    print("="*60)
    print("📊 测试结果总结:")
    print("  - 轻度遮挡: ./test_results_light_occlusion/")
    print("  - 中等遮挡: ./test_results_medium_occlusion/")
    print("  - 重度遮挡: ./test_results_heavy_occlusion/")
    print("  - 灰色遮挡: ./test_results_gray_occlusion/")
    print("\n💡 建议:")
    print("  1. 查看不同遮挡强度下的恢复效果")
    print("  2. 对比 05_Restored_PnPFlow_XX.png 和 01_HR_GT_XX.png")
    print("  3. 观察模型在遮挡区域的修复能力")
    print("  4. 比较不同遮挡类型（黑色 vs 灰色）的恢复差异")

if __name__ == "__main__":
    main()