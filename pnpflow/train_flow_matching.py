# Code adapted from
#
# Tong, A., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Fatras, K., ... & Bengio, Y. (2023).
# Improving and generalizing flow-based generative models with minibatch optimal transport.
# arXiv preprint arXiv:2302.00482.
# (https://github.com/atong01/conditional-flow-matching)
#
# Chemseddine, J., Hagemann, P., Wald, C., & Steidl, G. (2024).
# Conditional Wasserstein Distances with Applications in Bayesian OT Flow Matching.
# arXiv preprint arXiv:2403.18705.
# (https://github.com/JChemseddine/Conditional_Wasserstein_Distances/blob/main/utils/utils_FID.py)

import torch
import os
import skimage.io as io
import numpy as np
import torch
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import ot
from torchdiffeq import odeint_adjoint as odeint
from pnpflow.dataloaders import DataLoaders
import pnpflow.utils as utils
from pnpflow.models import InceptionV3
import pnpflow.fid_score as fs





class FLOW_MATCHING(object):

    def __init__(self, model, device, args):
        self.d = args.dim_image
        self.num_channels = args.num_channels
        self.device = device
        self.args = args
        self.lr = args.lr
        self.model = model  # 模型已经在main.py中移动到device并包装DataParallel

    def train_FM_model(self, train_loader, opt, num_epoch):
        tq = tqdm(range(num_epoch), desc='loss')
        for ep in tq:
            for iteration, (x, labels) in enumerate(train_loader):
                if x.size(0) == 0:
                    continue
                x = x.to(self.device)
                z = torch.randn(
                    x.shape[0],
                    self.num_channels,
                    self.d,
                    self.d,
                    device=self.device,
                    requires_grad=False)

                t1 = torch.rand(x.shape[0], 1, 1, 1, device=self.device)

                # compute coupling
                x0 = z.clone()
                x1 = x.clone()
                a, b = np.ones(len(x0)) / len(x0), np.ones(len(x0)) / len(x0)

                M = ot.dist(x0.view(len(x0), -1).cpu().data.numpy(),
                            x1.view(len(x1), -1).cpu().data.numpy())
                plan = ot.emd(a, b, M)
                p = plan.flatten()
                p = p / p.sum()
                choices = np.random.choice(
                    plan.shape[0] * plan.shape[1], p=p, size=len(x0), replace=True)
                i, j = np.divmod(choices, plan.shape[1])
                x0 = x0[i]
                x1 = x1[j]
                xt = t1 * x1 + (1 - t1) * x0
                # 使用像素平均的 MSE 以保证 loss 数值稳定 (≈0~5)
                # 确保时间步的维度正确
                t_input = t1.squeeze()
                if len(t_input.shape) == 0:  # 如果是标量，扩展为1D
                    t_input = t_input.unsqueeze(0)
                elif len(t_input.shape) > 1:  # 如果维度过多，压缩
                    t_input = t_input.view(-1)
                mse = (self.model(xt, t_input) - (x1 - x0))**2
                loss = mse.view(x.shape[0], -1).mean()  # mean over batch 和像素
                opt.zero_grad()
                loss.backward()
                opt.step()

                # save loss in txt file
                with open(self.save_path + 'loss_training.txt', 'a') as file:
                    file.write(
                        f'Epoch: {ep}, iter: {iteration}, Loss: {loss.item()}\n')

            # save samples, plot them, and compute FID on small dataset
            self.sample_plot(x, ep)
            if ep % 5 == 0:
                # save model
                model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                torch.save(model_state, self.model_path + 'model_{}.pt'.format(ep))
                # NOTE: 已请求停止PSNR评估，避免额外显存占用
                # 若需恢复，请取消以下注释并调整 num_samples 等参数
                # psnr_value = self.compute_psnr(64)
                # with open(self.save_path + 'psnr.txt', 'a') as file:
                #     file.write(f'Epoch: {ep}, PSNR: {psnr_value:.4f}\n')

    def apply_flow_matching(self, NO_samples):
        self.model.eval()
        with torch.no_grad():
            model_class = cnf(self.model)
            latent = torch.randn(
                NO_samples,
                self.num_channels,
                self.d,
                self.d,
                device=self.device,
                requires_grad=False)
            z_t = odeint(model_class, latent,
                         torch.tensor([0.0, 1.0]).to(self.device),
                         atol=1e-5,
                         rtol=1e-5,
                         method='dopri5',
                         )
            x = z_t[-1].detach()
        self.model.train()
        return x

    def sample_plot(self, x, ep=None):
        try:
            os.makedirs(self.save_path + 'results_samplings/')
        except BaseException:
            pass

        reco = utils.postprocess(self.apply_flow_matching(16), self.args)
        utils.save_samples(reco.detach().cpu(), x[:16].cpu(), self.save_path + 'results_samplings/' +
                           'samplings_ep_{}.pdf'.format(ep), self.args)

        # check the plots by saving training samples
        if ep == 0:
            gt = x[:16]
            gt = utils.postprocess(gt, self.args)
            utils.save_samples(gt.detach().cpu(), gt.detach().cpu(), self.save_path + 'results_samplings/' +
                               'train_samples_ep_{}.pdf'.format(ep), self.args)

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
    
    def compute_fast_fid(self, num_samples):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx]).to(self.device)
        data_v = next(iter(self.full_train_set))
        gt = data_v[0].to(self.device)[:num_samples]
        gt = gt.permute(0, 2, 3, 1).cpu().numpy()
        if gt.shape[-1] == 1:
            gt = np.concatenate([gt, gt, gt], axis=-1)
        gt = np.transpose(gt, axes=(0, 3, 1, 2))
        batch_size = 50
        m1, s1 = fs.calculate_activation_statistics(
            gt, model, batch_size, 2048, self.device)

        samples = torch.empty(0).to(self.device)
        n_iter = 50
        for i in range(n_iter):
            samples = torch.cat([samples.cpu(), self.apply_flow_matching(
                num_samples // n_iter).cpu()], dim=0)
        gen = torch.clip(samples.permute(0, 2, 3, 1), 0, 1).cpu().numpy()
        if gen.shape[-1] == 1:
            gen = np.concatenate([gen, gen, gen], axis=-1)
        gen = np.transpose(gen, axes=(0, 3, 1, 2))
        m2, s2 = fs.calculate_activation_statistics(
            gen, model, batch_size, 2048, self.device)
        fid_value = fs.calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value

    def train(self, data_loaders):

        self.save_path = self.args.root + \
            'results/{}/ot/'.format(
                self.args.dataset)
        try:
            os.makedirs(self.save_path)
        except BaseException:
            pass

        self.model_path = self.args.root + \
            'model/{}/ot/'.format(
                self.args.dataset)
        try:
            os.makedirs(self.model_path)
        except BaseException:
            pass

        # load model
        train_loader = data_loaders['train']

        # load full dataset on cpu to evaluate PSNR
        full_data = DataLoaders(self.args.dataset, 64, 64)  # 减少样本数量，只用于PSNR计算
        self.full_train_set = full_data.load_data()['train']

        # create txt file for storing all information about model
        with open(self.save_path + 'model_info.txt', 'w') as file:
            file.write(f'PARAMETERS\n')
            file.write(
                f'Number of parameters: {sum(p.numel() for p in self.model.parameters())}\n')
            file.write(f'Number of epochs: {self.args.num_epoch}\n')
            file.write(f'Batch size: {self.args.batch_size_train}\n')
            file.write(f'Learning rate: {self.lr}\n')

        # start training
        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.train_FM_model(train_loader, opt, num_epoch=self.args.num_epoch)

        # save final model
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        torch.save(model_state, self.model_path + 'model_final.pt')


class cnf(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        with torch.no_grad():
            # z = self.model(x, t.squeeze())
            z = self.model(x, t.repeat(x.shape[0]))
        return z
