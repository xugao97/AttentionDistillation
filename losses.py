import math

import numpy as np
import torch
import torch.nn.functional as F

loss_fn = torch.nn.L1Loss()


def ad_loss(
    q_list, ks_list, vs_list, self_out_list, scale=1, source_mask=None, target_mask=None
):
    loss = 0
    attn_mask = None
    for q, ks, vs, self_out in zip(q_list, ks_list, vs_list, self_out_list):
        if source_mask is not None and target_mask is not None:
            w = h = int(np.sqrt(q.shape[2]))
            mask_1 = torch.flatten(F.interpolate(source_mask, size=(h, w)))
            mask_2 = torch.flatten(F.interpolate(target_mask, size=(h, w)))
            attn_mask = mask_1.unsqueeze(0) == mask_2.unsqueeze(1)
            attn_mask=attn_mask.to(q.device)

        target_out = F.scaled_dot_product_attention(
            q * scale,
            torch.cat(torch.chunk(ks, ks.shape[0]), 2).repeat(q.shape[0], 1, 1, 1),
            torch.cat(torch.chunk(vs, vs.shape[0]), 2).repeat(q.shape[0], 1, 1, 1),
            attn_mask=attn_mask
        )
        loss += loss_fn(self_out, target_out.detach())
    return loss



def q_loss(q_list, qc_list):
    loss = 0
    for q, qc in zip(q_list, qc_list):
        loss += loss_fn(q, qc.detach())
    return loss

# weight = 200
def qk_loss(q_list, k_list, qc_list, kc_list):
    loss = 0
    for q, k, qc, kc in zip(q_list, k_list, qc_list, kc_list):
        scale_factor = 1 / math.sqrt(q.size(-1))
        self_map = torch.softmax(q @ k.transpose(-2, -1) * scale_factor, dim=-1)
        target_map = torch.softmax(qc @ kc.transpose(-2, -1) * scale_factor, dim=-1)
        loss += loss_fn(self_map, target_map.detach())
    return loss

# weight = 1
def qkv_loss(q_list, k_list, vc_list, c_out_list):
    loss = 0
    for q, k, vc, target_out in zip(q_list, k_list, vc_list, c_out_list):
        self_out = F.scaled_dot_product_attention(q, k, vc)
        loss += loss_fn(self_out, target_out.detach())
    return loss
