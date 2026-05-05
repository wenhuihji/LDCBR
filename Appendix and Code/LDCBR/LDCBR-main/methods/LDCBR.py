# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def _normalize_dist(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y = torch.clamp(y, min=0.0)
    y = y / (y.sum(dim=1, keepdim=True) + eps)
    y = torch.clamp(y, min=eps)
    y = y / (y.sum(dim=1, keepdim=True) + eps)
    return y


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(x + self.net(x)))


class HyperNetwork(nn.Module):
    def __init__(self, num_feature: int, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.mask_emb = nn.Linear(num_classes, hidden_dim)
        self.feat_emb = nn.Linear(num_feature, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes * 2),
        )

    def forward(self, x, y_missing):
        mask = (y_missing > 0).float()
        h_m = F.relu(self.mask_emb(mask))
        h_x = F.relu(self.feat_emb(x))
        ctx = torch.cat([h_x, h_m], dim=1)
        params = self.net(ctx)
        w_a, w_b = params.chunk(2, dim=1)
        w_a = torch.sigmoid(w_a)
        w_b = torch.sigmoid(w_b)
        return w_a, w_b, mask


class ComponentDecomposition(nn.Module):
    def __init__(self, num_classes: int, rank: int = 3):
        super().__init__()
        self.basis = nn.Parameter(torch.empty(rank, num_classes))
        nn.init.orthogonal_(self.basis)

    def forward(self, y):
        coeff = y @ self.basis.t()
        comp_a = coeff @ self.basis
        comp_b = y - comp_a
        return _normalize_dist(F.relu(comp_a)), _normalize_dist(F.relu(comp_b))


class AdaptiveFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=1)
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, h_x, h_a, h_b):
        concat_feat = torch.cat([h_x, h_a, h_b], dim=1)
        weights = self.gate_net(concat_feat)
        w_x, w_a, w_b = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        h_fused = w_x * h_x + w_a * h_a + w_b * h_b
        return self.out_proj(h_fused), w_x


class LDCBR(nn.Module):
    def __init__(
        self,
        num_feature,
        num_classes,
        hidden_dim=256,
        lr=1e-3,
        weight_decay=1e-4,
        device="cuda",
        decomp_rank=4,
        kl_weight=1.0,
        warmup_steps=500,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.base_lr = lr
        self.kl_weight = kl_weight
        self.warmup_steps = warmup_steps

        self.lambda_smooth = 1e-3
        self.lambda_basis = 1e-4

        self.hyper = HyperNetwork(num_feature, num_classes, hidden_dim)
        self.decomp = ComponentDecomposition(num_classes, decomp_rank)

        self.enc_x = nn.Sequential(
            nn.Linear(num_feature, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            ResBlock(hidden_dim, dropout=0.3),
            ResBlock(hidden_dim, dropout=0.3)
        )

        self.enc_y = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            ResBlock(hidden_dim, dropout=0.2)
        )

        self.fusion = AdaptiveFusion(hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        self.register_buffer("step", torch.tensor(0))
        self.to(self.device)

    def forward(self, x, y_missing):
        w_a, w_b, mask = self.hyper(x, y_missing)
        y_comp = _normalize_dist(y_missing * w_a + (1 - w_a) * w_b)
        c_a, c_b = self.decomp(y_comp)

        h_x = self.enc_x(x)
        h_a = self.enc_y(c_a)
        h_b = self.enc_y(c_b)

        h_fused, gate_w = self.fusion(h_x, h_a, h_b)
        logits = self.decoder(h_fused)

        return logits, y_comp, gate_w

    def train_step(self, x, y_missing, y_full=None):
        self.train()
        x = x.to(self.device)
        y_missing = y_missing.to(self.device)

        logits, y_comp, gate_w = self.forward(x, y_missing)

        log_pred = F.log_softmax(logits, dim=1)
        pred = torch.exp(log_pred)

        eps = 1e-8
        obs_mask = (y_missing > 0).float()
        target_dist = obs_mask / (obs_mask.sum(dim=1, keepdim=True) + eps)
        loss_obs = -(target_dist * log_pred).sum(dim=1).mean()

        loss_kl = F.kl_div(
            log_pred,
            y_comp.detach(),
            reduction='batchmean'
        )

        basis = self.decomp.basis
        gram = basis @ basis.t()
        loss_basis = F.mse_loss(gram, torch.eye(gram.shape[0], device=self.device))

        loss_smooth = (y_comp ** 2).sum(dim=1).mean()

        loss = (
            loss_obs
            + self.kl_weight * loss_kl
            + 0.1 * loss_basis
            + 0.01 * loss_smooth
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        if self.step < self.warmup_steps:
            scale = float(self.step + 1) / float(self.warmup_steps)
            for g in self.optimizer.param_groups:
                g["lr"] = scale * self.base_lr
        else:
            self.scheduler.step()
        self.step += 1

        return float(loss.item())

    @torch.no_grad()
    def get_result(self, test_loader):
        self.eval()
        preds, labels = [], []
        for x, y_missing, y_full in test_loader:
            x = x.to(self.device)
            y_in = y_missing.to(self.device)
            logits, _, _ = self.forward(x, y_in)
            preds.append(F.softmax(logits, dim=1).cpu().numpy())
            labels.append(y_full.numpy())
        return np.concatenate(preds), np.concatenate(labels)
