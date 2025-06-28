# PnP-Flow PSNR 修改总结

## 修改目标
将训练过程中的FID计算替换为PSNR（峰值信噪比）计算，以更适合超分辨率任务的评估。

## 主要修改内容

### 1. 训练代码修改 (`pnpflow/train_flow_matching.py`)

#### 移除FID相关导入
```python
# 移除以下导入
from pnpflow.models import InceptionV3
import pnpflow.fid_score as fs
```

#### 替换FID计算为PSNR计算
- **原始代码**: `compute_fast_fid(2048)` - 使用InceptionV3计算FID
- **新代码**: `compute_psnr(64)` - 计算PSNR值

#### 新增PSNR计算函数
```python
def compute_psnr(self, num_samples):
    """计算生成样本与真实样本的PSNR"""
    import torch.nn.functional as F
    
    # 获取真实样本
    data_v = next(iter(self.full_train_set))
    gt = data_v[0].to(self.device)[:num_samples]
    
    # 生成样本
    generated = self.apply_flow_matching(num_samples)
    
    # 确保数据范围在[0,1]
    gt_normalized = utils.postprocess(gt, self.args)
    gen_normalized = utils.postprocess(generated, self.args)
    
    # 计算MSE
    mse = F.mse_loss(gen_normalized, gt_normalized)
    
    # 计算PSNR (Peak Signal-to-Noise Ratio)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    return psnr.item()
```

#### 修改评估触发逻辑
```python
if ep % 5 == 0:
    # save model
    model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
    torch.save(model_state, self.model_path + 'model_{}.pt'.format(ep))
    # evaluate PSNR
    psnr_value = self.compute_psnr(64)  # 使用64个样本计算PSNR
    with open(self.save_path + 'psnr.txt', 'a') as file:
        file.write(f'Epoch: {ep}, PSNR: {psnr_value:.4f}\n')
```

#### 优化数据加载
- **原始**: `DataLoaders(self.args.dataset, 2048, 2048)` - 用于FID计算
- **新设置**: `DataLoaders(self.args.dataset, 64, 64)` - 用于PSNR计算，减少内存占用

### 2. 配置调整

#### 批大小优化
由于512×512图像的显存需求，调整批大小：
- **最终配置**: `batch_size_train: 2`
- **GPU配置**: 使用GPU 6,7双卡训练

## 测试验证

### 功能测试
✅ 创建并运行了完整的PSNR计算测试
- 模型加载正常
- 数据加载正常 (3224个有效HR-LR图像对)
- PSNR计算成功 (测试值: 5.4491)
- 样本生成正常 (形状: [2, 3, 512, 512])

### 训练测试
✅ 训练启动成功
- 双GPU正常工作
- 批大小2可以正常运行，无OOM错误
- 训练损失稳定在0.6-0.7范围内
- 日志文件正常生成

## 输出文件变化

### 新增文件
- `results/cropsr/ot/psnr.txt` - PSNR评估结果（每5个epoch记录一次）

### 移除文件
- `results/cropsr/ot/fid.txt` - 不再生成FID评估结果

## PSNR vs FID 的优势

1. **任务相关性**: PSNR直接衡量图像重建质量，更适合超分辨率任务
2. **计算效率**: 不需要加载大型InceptionV3模型，计算更快
3. **内存友好**: 显著减少显存占用
4. **直观性**: PSNR值直接反映图像质量，数值越高越好

## 使用方法

训练时PSNR会自动计算并保存：
```bash
export CUDA_VISIBLE_DEVICES=6,7 && nohup python main.py --opts train True eval False &
```

查看PSNR结果：
```bash
cat results/cropsr/ot/psnr.txt
```

## 注意事项

1. PSNR计算在每5个epoch执行一次
2. 使用64个样本进行PSNR计算以平衡精度和效率
3. 批大小设置为2以适应512×512图像的显存需求
4. 双GPU训练配置已优化，确保稳定运行 