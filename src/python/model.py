import torch
import numpy as np
import torch.utils.data

from vmamba2 import VSSBlock
from scipy.optimize import linear_sum_assignment
from layers import RMSNorm, RMSNorm2D, MaskDecoder


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, with_logits=False):
        super(FocalLoss, self).__init__()

        self._alpha = alpha
        self._gamma = gamma
        self._with_logits = with_logits

    def forward(self, inputs, targets):
        probs = inputs.sigmoid() if self._with_logits else inputs

        probs = torch.clamp(probs, 0.001, 0.999)
        pt = probs * targets + (1 - probs) * (1 - targets)
        ce_loss = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))

        focal_loss = ce_loss * torch.pow(1 - pt, self._gamma)
        if self._alpha >= 0:
            at = self._alpha * targets + (1 - self._alpha) * (1 - targets)
            focal_loss = at * focal_loss
        return focal_loss


class MultiClassFocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, with_logits=False):
        super(MultiClassFocalLoss, self).__init__()

        if alpha is None:
            alpha = [0.0, 0.280, 0.336, 0.120, 0.153, 0.112]  # [0.0, 0.280, 0.336, 0.120, 0.153, 0.112]
        else:
            assert len(alpha) == 6
        self._alpha = torch.as_tensor(alpha)[None, :, None]
        self._gamma = gamma
        self._with_logits = with_logits

    def forward(self, inputs, targets):
        alpha = torch.gather(torch.tile(self._alpha, [inputs.shape[0], 1, inputs.shape[2]]).to(targets.device), 1, targets[:, None]).squeeze()
        log_pt = torch.gather(torch.log_softmax(inputs, 1) if self._with_logits else torch.log(inputs), 1, targets[:, None]).squeeze()
        pt = torch.exp(log_pt)
        focal_loss = -alpha * torch.pow(1 - pt, self._gamma) * log_pt
        return focal_loss


class DiceSemimetricLoss(torch.nn.Module):
    def __init__(self, axis, with_logits=False):
        super(DiceSemimetricLoss, self).__init__()

        self._axis = axis
        self._with_logits = with_logits

    def forward(self, inputs, targets):
        probs = inputs.sigmoid() if self._with_logits else inputs

        negative = torch.sum(torch.abs(probs - targets), self._axis)
        positive = torch.sum(torch.abs(probs) + torch.abs(targets), self._axis)

        return 1 - (positive - negative) / (positive + 1e-3)


class MomentumL1Loss(torch.nn.Module):
    def __init__(self, with_transform=False):
        super(MomentumL1Loss, self).__init__()

        self._with_transform = with_transform

    def forward(self, inputs, targets):
        predicts = inputs if self._with_transform else calc_bbox(inputs)

        targets_extra = torch.nn.functional.pad(torch.where(predicts[..., 2] - targets[..., 2] > torch.pi, 2 * torch.pi, 0.0).unsqueeze(-1), (2, 1)).detach()
        predicts_extra = torch.nn.functional.pad(torch.where(targets[..., 2] - predicts[..., 2] > torch.pi, 2 * torch.pi, 0.0).unsqueeze(-1), (2, 1)).detach()

        targets = targets + targets_extra
        predicts = predicts + predicts_extra

        error = predicts / (targets + 1e-3)
        return torch.nn.functional.l1_loss(error, torch.ones_like(error).detach(), reduction="none").mean(-1)


class EIoULoss(torch.nn.Module):
    def __init__(self, with_transform=False):
        super(EIoULoss, self).__init__()

        self._with_transform = with_transform

    def forward(self, inputs, targets):
        inputs = inputs if self._with_transform else calc_bbox(inputs)

        w1, x1, y1, h1 = torch.split(inputs, 1, -1)
        w2, x2, y2, h2 = torch.split(targets, 1, -1)

        y1_extra = torch.where(y2 - y1 > torch.pi, 2 * torch.pi, 0.0)
        y2_extra = torch.where(y1 - y2 > torch.pi, 2 * torch.pi, 0.0)

        y1 = y1 + y1_extra
        y2 = y2 + y2_extra

        w = torch.min(x1 + w1 / 2, x2 + w2 / 2) - torch.max(x1 - w1 / 2, x2 - w2 / 2)
        h = torch.min(y1 + h1 / 2, y2 + h2 / 2) - torch.max(y1 - h1 / 2, y2 - h2 / 2)

        iou = torch.where(torch.logical_and(w > 0, h > 0), w * h / (w1 * h1 + w2 * h2 - w * h + 1e-3), 0)

        cw2 = torch.square(torch.max(x1 + w1 / 2, x2 + w2 / 2) - torch.min(x1 - w1 / 2, x2 - w2 / 2))
        ch2 = torch.square(torch.max(y1 + h1 / 2, y2 + h2 / 2) - torch.min(y1 - h1 / 2, y2 - h2 / 2))

        rho_w = torch.square(w1 - w2) / (cw2 + 1e-3)
        rho_h = torch.square(h1 - h2) / (ch2 + 1e-3)
        rho_d = (torch.square(x1 - x2) + torch.square(y1 - y2)) / (cw2 + ch2 + 1e-3)

        eiou_loss = 1 - iou + rho_d + rho_w + rho_h

        return eiou_loss.squeeze(-1)


class HungarianMatcher(torch.nn.Module):
    def __init__(self, cost_mask=(0.25, 0.75), cost_class=1.0, cost_momentum=(0.8, 0.2), with_logits=False):
        super().__init__()

        self._cost_mask = cost_mask
        self._cost_class = cost_class
        self._cost_momentum = cost_momentum
        self._with_logits = with_logits

        self._l1_loss = MomentumL1Loss()
        self._eiou_loss = EIoULoss()
        self._focal_loss = FocalLoss(with_logits=with_logits)
        self._dice_semimetric_loss = DiceSemimetricLoss((2, 3), with_logits)

    @torch.no_grad()
    def _bipartite_matching(self, inputs, targets):
        pred_prob, pred_mask, pred_momentum = inputs
        target_class, target_mask, target_momentum = targets

        batch_size = pred_prob.shape[0]
        pred_prob = pred_prob.softmax(1) if self._with_logits else pred_prob
        pred_mask = torch.nn.functional.interpolate(pred_mask, target_mask.shape[-2:], mode="bilinear")

        indices = []
        for b in range(batch_size):
            cost_class = -pred_prob[b, target_class[b]].transpose(0, 1)
            cost_l1 = self._l1_loss(pred_momentum[b, :, None], target_momentum[b, None])
            cost_eiou = self._eiou_loss(pred_momentum[b, :, None], target_momentum[b, None])
            cost_dice_mask = self._dice_semimetric_loss(pred_mask[b, :, None], target_mask[b, None])
            cost_focal_mask = self._focal_loss(pred_mask[b, :, None], target_mask[b, None]).mean((2, 3))
            cost = self._cost_mask[0] * cost_focal_mask + self._cost_mask[1] * cost_dice_mask + self._cost_class * cost_class + self._cost_momentum[0] * cost_l1 + self._cost_momentum[1] * cost_eiou
            indices.append(linear_sum_assignment(torch.where(target_class[b, None] > 0, cost, 1e5).detach().cpu()))
        indices = torch.as_tensor(np.transpose(np.array(indices), [0, 2, 1]))
        return indices

    @torch.no_grad()
    def forward(self, inputs, targets):
        return self._bipartite_matching(inputs, targets)


class LhcSAM(torch.nn.Module):
    def __init__(self, image_size, patch_size, n_block=15, encode_dim=256):
        super(LhcSAM, self).__init__()

        self._n_patch = image_size // patch_size
        self._d_model = patch_size * patch_size * 3
        self._encode_dim = encode_dim

        self._patch_embedding = torch.nn.Sequential(torch.nn.Conv2d(6, self._d_model, patch_size, patch_size), RMSNorm2D(self._d_model))
        self._position_embedding = torch.nn.Parameter(torch.rand(self._d_model, self._n_patch, self._n_patch))
        self._image_encoder = torch.nn.Sequential(*[VSSBlock(self._d_model, norm_layer=RMSNorm, channel_first=False) for _ in range(n_block)])
        self._neck = torch.nn.Sequential(torch.nn.Conv2d(self._d_model, encode_dim, 1), RMSNorm2D(encode_dim),
                                         torch.nn.Conv2d(encode_dim, encode_dim, 3, 1, 1), RMSNorm2D(encode_dim))
        self._mask_decoder = MaskDecoder(encode_dim)

    def forward(self, inputs):
        image, state = inputs
        patch = self._patch_embedding(image) + self._position_embedding
        image_embedding = self._neck(self._image_encoder(patch.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

        return self._mask_decoder([image_embedding, state])


def calc_bbox(inputs):
    pt, rap, phi, m = torch.split(inputs, 1, -1)

    pt, m = pt.clamp_max(10).exp(), m.clamp_max(10).exp()
    rap, phi = rap.sigmoid() * 2 * torch.pi, phi.sigmoid() * 2 * torch.pi

    return torch.concatenate([pt, rap, phi, m], -1)


def calc_mask_jaccard_metric(pred_mask, target_mask, with_logits=False):
    pred_mask = pred_mask.sigmoid() if with_logits else pred_mask

    negative = torch.sum(torch.abs(pred_mask - target_mask), (-2, -1))
    positive = torch.sum(torch.abs(pred_mask) + torch.abs(target_mask), (-2, -1))

    return (positive - negative) / (positive + negative + 1e-3)


def calc_momentum_iou(pred_momentum, target_momentum, with_transform=False):
    pred_momentum = pred_momentum if with_transform else calc_bbox(pred_momentum)

    w1, x1, y1, h1 = torch.split(pred_momentum, 1, -1)
    w2, x2, y2, h2 = torch.split(target_momentum, 1, -1)

    y1_extra = torch.where(y2 - y1 > torch.pi, 2 * torch.pi, 0.0)
    y2_extra = torch.where(y1 - y2 > torch.pi, 2 * torch.pi, 0.0)

    y1 = y1 + y1_extra
    y2 = y2 + y2_extra

    w = torch.min(x1 + w1 / 2, x2 + w2 / 2) - torch.max(x1 - w1 / 2, x2 - w2 / 2)
    h = torch.min(y1 + h1 / 2, y2 + h2 / 2) - torch.max(y1 - h1 / 2, y2 - h2 / 2)

    iou = torch.where(torch.logical_and(w > 0, h > 0), w * h / (w1 * h1 + w2 * h2 - w * h + 1e-3), 0)
    return iou.squeeze(-1)
