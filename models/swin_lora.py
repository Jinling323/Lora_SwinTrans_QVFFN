"""Low-rank updates for Swin attention and the MAN Transformer FFN."""

import torch
from torch import nn
from torch.nn.utils import parametrize


class QVLoRA(nn.Module):
    def __init__(self, dim, rank, alpha):
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError('LoRA rank and alpha must be positive')
        self.scaling = alpha / rank
        self.lora_A_q = nn.Parameter(torch.empty(rank, dim))
        self.lora_B_q = nn.Parameter(torch.zeros(dim, rank))
        self.lora_A_v = nn.Parameter(torch.empty(rank, dim))
        self.lora_B_v = nn.Parameter(torch.zeros(dim, rank))
        nn.init.kaiming_uniform_(self.lora_A_q, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.lora_A_v, a=5 ** 0.5)

    def forward(self, weight):
        q_update = self.lora_B_q @ self.lora_A_q
        v_update = self.lora_B_v @ self.lora_A_v
        return weight + self.scaling * torch.cat(
            (q_update, torch.zeros_like(q_update), v_update), dim=0)


class LinearLoRA(nn.Module):
    """LoRA parametrization for a regular ``nn.Linear`` weight."""

    def __init__(self, in_features, out_features, rank, alpha):
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError('LoRA rank and alpha must be positive')
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, weight):
        return weight + self.scaling * (self.lora_B @ self.lora_A)


def add_swin_qv_lora(features, rank, alpha):
    """Attach LoRA to every W-MSA/SW-MSA block found in Swin features."""
    blocks = [module for module in features.modules()
              if hasattr(module, 'attn') and hasattr(module.attn, 'qkv')]
    if not blocks:
        raise ValueError('No Swin attention blocks found')
    for block in blocks:
        qkv = block.attn.qkv
        if not isinstance(qkv, nn.Linear) or qkv.out_features != 3 * qkv.in_features:
            raise ValueError('Swin attention QKV must be a packed Linear projection')
        if parametrize.is_parametrized(qkv, 'weight'):
            raise ValueError('LoRA is already attached to a Swin attention block')
        parametrize.register_parametrization(
            qkv, 'weight', QVLoRA(qkv.in_features, rank, alpha))
    return len(blocks)


def add_man_ffn_lora(encoder, rank, alpha):
    """Attach LoRA to both FFN projections in every MAN encoder layer."""
    projections = []
    for layer in encoder.layers:
        for name in ('linear1', 'linear2'):
            linear = getattr(layer, name, None)
            if not isinstance(linear, nn.Linear):
                raise ValueError(
                    'MAN Transformer layer {} must be nn.Linear'.format(name))
            if parametrize.is_parametrized(linear, 'weight'):
                raise ValueError(
                    'LoRA is already attached to MAN Transformer {}'.format(name))
            projections.append(linear)

    if not projections:
        raise ValueError('No MAN Transformer FFN projections found')

    for linear in projections:
        parametrize.register_parametrization(
            linear,
            'weight',
            LinearLoRA(linear.in_features, linear.out_features, rank, alpha),
        )
    return len(projections)
