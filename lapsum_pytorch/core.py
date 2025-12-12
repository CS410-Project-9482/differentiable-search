"""
LapSum Core Implementation
==========================
Differentiable Log-Space Soft Top-K Operator.

ATTRIBUTION NOTICE:
This module is a direct port of the logic found in `tutorial_notebooks/log-soft_topk.ipynb`
from the official LapSum repository: https://github.com/gmum/LapSum

Reference:
    "LapSum: One Method to Differentiate Them All: Ranking, Sorting and Top-k Selection"
    (ICML 2025)
"""

import torch
from math import log

class LogSoftTopK(torch.autograd.Function):
    """
    Autograd function for the Log-Space Soft Top-K operator.
    Solves the convex dual problem to find the optimal threshold 'b'.
    """

    @staticmethod
    def _solve(s, t, a, b, e):
        z = torch.abs(e) + torch.sqrt(e**2 + a * b * torch.exp(s - t))
        ab = torch.where(e > 0, a, b)
        return torch.where(
            e > 0, 
            t + torch.log(z) - torch.log(ab), 
            s - torch.log(z) + torch.log(ab)
        )

    @staticmethod
    def forward(ctx, r, k, alpha, descending=False):
        assert r.shape[0] == k.shape[0], "k must have same batch size as r"
        batch_size, num_dim = r.shape
        x = torch.empty_like(r, requires_grad=False)

        def finding_b():
            scaled = torch.sort(r, dim=1)[0]
            scaled.div_(alpha)
            eB = torch.logcumsumexp(scaled, dim=1)
            eB.sub_(scaled).exp_()
            torch.neg(scaled, out=x)
            eA = torch.flip(x, dims=(1,))
            torch.logcumsumexp(eA, dim=1, out=x)
            idx = torch.arange(start=num_dim - 1, end=-1, step=-1, device=x.device)
            torch.index_select(x, 1, idx, out=eA)
            eA.add_(scaled).exp_()
            row = torch.arange(1, 2 * num_dim + 1, 2, device=r.device)
            torch.add(torch.add(eA, eB, alpha=-1, out=x), row.view(1, -1), out=x)
            w = (k if descending else num_dim - k).unsqueeze(1)
            i = torch.searchsorted(x, 2 * w)
            m = torch.clamp(i - 1, 0, num_dim - 1)
            n = torch.clamp(i, 0, num_dim - 1)
            b = LogSoftTopK._solve(
                scaled.gather(1, m),
                scaled.gather(1, n),
                torch.where(i < num_dim, eA.gather(1, n), 0),
                torch.where(i > 0, eB.gather(1, m), 0),
                w - i,
            )
            return b

        b = finding_b()
        sign = -1 if descending else 1
        torch.div(r, alpha * sign, out=x)
        x.sub_(sign * b)
        sign_x = x > 0
        qx = torch.relu(x).neg_().exp_().mul_(-0.5).add_(1)
        ctx.save_for_backward(x, qx, r)
        ctx.alpha = alpha
        ctx.sign = sign
        log_p = torch.where(sign_x, torch.log(qx), x.sub(log(2)))
        return log_p

    @staticmethod
    def backward(ctx, grad_output):
        x, qx, r = ctx.saved_tensors
        alpha = ctx.alpha
        sign = ctx.sign
        x.abs_().neg_()
        grad_r = torch.softmax(x, dim=1)
        x.exp_()
        grad_k = torch.sum(x, dim=1).mul_(0.5)
        qx.reciprocal_().sub_(1)
        qx.mul_(grad_output)
        wsum = qx.sum(dim=1, keepdim=True)
        grad_k.reciprocal_().mul_(wsum.squeeze(1)).mul_(abs(sign))
        grad_r.mul_(wsum).sub_(qx).mul_(-sign / alpha)
        x.copy_(r).mul_(grad_r)
        grad_alpha = torch.sum(x).div_(-alpha)
        return grad_r, grad_k, grad_alpha, None

def log_soft_top_k(r, k, alpha, descending=False):
    return LogSoftTopK.apply(r, k, alpha, descending)
