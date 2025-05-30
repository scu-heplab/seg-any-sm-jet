import os
import torch
import pickle
import numpy as np
import torch.utils.data

from vmamba2 import VSSBlock
from collections import namedtuple
from dataset_loader import DatasetLoader
from scipy.optimize import linear_sum_assignment
from layers import RMSNorm, RMSNorm2D, MaskDecoder
from torch.utils.data.distributed import DistributedSampler

MAPRecord = namedtuple('MAPRecord', ('cls', 'conf', 'iou', 'unique'))
KINRecord = namedtuple('KINRecord', ('pred_kin', 'target_kin', 'unique'))
ROCRecord = namedtuple('ROCRecord', ('pred_prob', 'target_cls', 'unique'))


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
            alpha = [0.0, 0.280, 0.336, 0.120, 0.153, 0.112] # [0.0, 0.280, 0.336, 0.120, 0.153, 0.112]
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


def init():
    from torch.distributed import init_process_group

    os.environ["NCCL_SOCKET_IFNAME"] = "lo"

    init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def train(batch_size=16, epochs=10, pretrain: str = None):
    rank = int(os.environ["RANK"])
    worker = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    print("rank [%d], worker [%d]" % (rank, worker))

    lhc = LhcSAM(315, 5).cuda()
    optimizer = torch.optim.AdamW(lhc.parameters(), 1.25e-4 * worker)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1.25e-4 * worker / 100)

    if rank == 0 and local_rank == 0:
        if os.path.exists(pretrain):
            state = torch.load(pretrain, weights_only=True)
            lhc.load_state_dict(state["lhc"])
            # optimizer.load_state_dict(state["optimizer"])
            # scheduler.load_state_dict(state["scheduler"])
            print("load epoch = [%d], lr = [%.6f]" % (state["epoch"], optimizer.param_groups[0]["lr"]))
            # start = torch.as_tensor(state["epoch"]).cuda()
            start = torch.as_tensor(0).cuda()
        else:
            start = torch.as_tensor(0).cuda()
    else:
        start = torch.as_tensor(0).cuda()

    lhc = torch.nn.parallel.DistributedDataParallel(lhc, device_ids=[local_rank], output_device=local_rank)

    l1_loss = MomentumL1Loss()
    eiou_loss = EIoULoss()
    focal_loss = FocalLoss(with_logits=True)
    matcher = HungarianMatcher(with_logits=True)
    mutil_class_loss = MultiClassFocalLoss(with_logits=True)
    dice_semimetric_loss = DiceSemimetricLoss((2, 3), True)

    torch.distributed.broadcast(start, 0)
    dataset = DatasetLoader("train", n_mix=50)
    dataset = torch.utils.data.DataLoader(dataset, batch_size, False, num_workers=20, sampler=DistributedSampler(dataset), drop_last=True)
    for i in range(start, epochs):
        step = np.zeros((5,))
        mean_kin = np.zeros((5,))
        mean_iou = np.zeros((5,))
        mean_conf_loss = np.zeros((1,))
        mean_mask_loss = np.zeros((1,))
        mean_class_loss = np.zeros((1,))
        mean_momentum_loss = np.zeros((1,))

        if rank == 0 and local_rank == 0:
            print("epoch: %d/%d, lr: %.6f" % (i + 1, epochs, optimizer.param_groups[0]["lr"]))
        elif local_rank == 0:
            print("epoch: %d/%d" % (i + 1, epochs))

        dataset.sampler.set_epoch(i)
        for j, data in enumerate(dataset):
            image, prompt, mask_label, class_label, momentum_label = [dat.cuda() for dat in data]
            pred_conf, pred_mask, pred_class, pred_momentum = lhc([image, prompt])

            batch = torch.arange(batch_size)[:, None]
            indices = matcher([pred_class.transpose(1, 2), pred_mask, pred_momentum], [class_label, mask_label, momentum_label])

            pred_conf = pred_conf[batch, indices[..., 0]]
            pred_mask = torch.nn.functional.interpolate(pred_mask[batch, indices[..., 0]], mask_label.shape[-2:], mode="bilinear")
            pred_class = pred_class[batch, indices[..., 0]]
            pred_momentum = pred_momentum[batch, indices[..., 0]]

            target_mask = mask_label[batch, indices[..., 1]]
            target_class = class_label[batch, indices[..., 1]]
            target_momentum = momentum_label[batch, indices[..., 1]]

            ignore = torch.where(target_class != 0, 1.0, 0.0)

            kin = calc_momentum_iou(pred_momentum.detach(), target_momentum)
            iou = calc_mask_jaccard_metric(pred_mask.detach(), target_mask, True)
            cls = pred_class.detach().softmax(-1)[batch, torch.arange(32)[None], target_class]
            conf_loss = torch.nn.functional.l1_loss(pred_conf, (kin + iou + cls) / 3 * ignore, reduction="mean")

            dl = (dice_semimetric_loss(pred_mask, target_mask) * ignore).sum(1) / (ignore.sum(1) + 1e-3)
            fl = (focal_loss(pred_mask, target_mask) * ignore[..., None, None]).mean((2, 3)).sum(1) / (ignore.sum(1) + 1e-3)
            mask_loss = 0.25 * fl.mean() + 0.75 * dl.mean()

            # class_loss = torch.nn.functional.cross_entropy(pred_class.transpose(1, 2), target_class)
            # class_loss = mutil_class_loss(pred_class.transpose(1, 2), target_class).sum(1).mean()
            class_loss = ((mutil_class_loss(pred_class.transpose(1, 2), target_class) * ignore).sum(1) / (ignore.sum(1) + 1e-3)).mean()

            # momentum_loss = ((eiou_loss(pred_momentum, target_momentum) * ignore).sum(1) / (ignore.sum(1) + 1e-3)).mean()
            l1 = (l1_loss(pred_momentum, target_momentum) * ignore).sum(1) / (ignore.sum(1) + 1e-3)
            el = (eiou_loss(pred_momentum, target_momentum) * ignore).sum(1) / (ignore.sum(1) + 1e-3)
            momentum_loss = 0.8 * l1.mean() + 0.2 * el.mean()

            loss = conf_loss / (conf_loss.detach() + 1e-5) + mask_loss / (mask_loss.detach() + 1e-5) + class_loss / (class_loss.detach() + 1e-5) + momentum_loss / (momentum_loss.detach() + 1e-5)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lhc.parameters(), 1.0)
            optimizer.step()

            mean_conf_loss += (conf_loss.detach().cpu().numpy() - mean_conf_loss) / (j + 1)
            mean_mask_loss += (mask_loss.detach().cpu().numpy() - mean_mask_loss) / (j + 1)
            mean_class_loss += (class_loss.detach().cpu().numpy() - mean_class_loss) / (j + 1)
            mean_momentum_loss += (momentum_loss.detach().cpu().numpy() - mean_momentum_loss) / (j + 1)

            kin_tmp = np.zeros((6,))
            iou_tmp = np.zeros((6,))
            index = torch.where(target_class != 0)
            np.add.at(kin_tmp, target_class[index].cpu().numpy(), kin[index].cpu().numpy())
            np.add.at(iou_tmp, target_class[index].cpu().numpy(), iou[index].cpu().numpy())
            np.divide.at(kin_tmp, *np.unique(target_class[index].cpu().numpy(), return_counts=True))
            np.divide.at(iou_tmp, *np.unique(target_class[index].cpu().numpy(), return_counts=True))

            step_tmp = np.zeros((6,))
            np.add.at(step_tmp, *np.unique(target_class[index].cpu().numpy(), return_counts=True))
            step += np.where(step_tmp[1:] > 0, 1, 0)
            mean_kin += (np.where(step_tmp[1:] > 0, kin_tmp[1:], mean_kin) - mean_kin) / (step + 1e-3)
            mean_iou += (np.where(step_tmp[1:] > 0, iou_tmp[1:], mean_iou) - mean_iou) / (step + 1e-3)

            if rank == 0 and local_rank == 0:
                print(f"step: {j + 1}/{len(dataset)}, conf_loss: {mean_conf_loss[0].round(5)}, mask_loss: {mean_mask_loss[0].round(5)}, class_loss: {mean_class_loss[0].round(5)}, momentum_loss: {mean_momentum_loss[0].round(5)}      ")
                print(f"kin_iou:  [t: {mean_kin[0].round(5)}, h: {mean_kin[1].round(5)}, w/z: {mean_kin[2].round(5)}, b: {mean_kin[3].round(5)}, j: {mean_kin[4].round(5)}]      ")
                print(f"mask_iou: [t: {mean_iou[0].round(5)}, h: {mean_iou[1].round(5)}, w/z: {mean_iou[2].round(5)}, b: {mean_iou[3].round(5)}, j: {mean_iou[4].round(5)}]      ")
                if j + 1 != len(dataset):
                    print("\033[F" * 3, end="")
            elif local_rank == 0:
                print(f"step: {j + 1}/{len(dataset)}")
                if j + 1 != len(dataset):
                    print("\033[F", end="")
        scheduler.step()

        if rank == 0 and local_rank == 0:
            torch.save({"lhc": lhc.module.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": i + 1}, "./weights/state.pth")


@torch.no_grad()
def prediction_process(predicts, targets, conf_threshold=0.3):
    target_mask, target_class, target_momentum = targets
    predict_conf, predict_mask, predict_class, predict_momentum = predicts

    predict_mask = torch.nn.functional.interpolate(predict_mask, target_mask.shape[-2:], mode="bilinear")

    mask_iou = calc_mask_jaccard_metric(predict_mask[:, :, None], target_mask[:, None], True)
    mask_iou = torch.where(torch.as_tensor(torch.unsqueeze(target_class, 1) != 0), mask_iou, -torch.inf)

    iou_ignore = torch.where(mask_iou.amax(-1) >= 0.5, 1.0, 0.0)
    conf_ignore = torch.where(predict_conf > conf_threshold, 1.0, 0.0)

    batch_index, component_index = torch.where(iou_ignore * conf_ignore > 0)

    order = mask_iou.amax(-1)[batch_index, component_index].argsort(descending=True)

    batch_index = batch_index[order]
    component_index = component_index[order]
    target_match_index = mask_iou.argmax(-1)[batch_index, component_index]

    unique_match, unique_match_index = torch.stack([batch_index, target_match_index], -1).unique(dim=0, return_inverse=True)
    unique_match_mask = torch.as_tensor([1 if index not in unique_match_index[:i] else 0 for i, index in enumerate(unique_match_index)], dtype=torch.int32)

    roc_record = [ROCRecord(prob, cls, mask) for prob, cls, mask in zip(predict_class[batch_index, component_index].softmax(-1).cpu().numpy(), target_class[batch_index, target_match_index].cpu().numpy(), unique_match_mask.cpu().numpy())]
    kin_record = [KINRecord(pred, target, mask) for pred, target, mask in zip(calc_bbox(predict_momentum[batch_index, component_index]).cpu().numpy(), target_momentum[batch_index, target_match_index].cpu().numpy(), unique_match_mask.cpu().numpy())]

    # mAP预计算，同类的预测框与GT框进行匹配 #

    mask_iou = torch.where(torch.argmax(predict_class, -1, keepdim=True) == torch.unsqueeze(target_class, 1), mask_iou, -torch.inf)

    iou_ignore = torch.where(mask_iou.amax(-1) >= 0.0, 1.0, 0.0)

    batch_index, component_index = torch.where(iou_ignore * conf_ignore > 0)

    order = mask_iou.amax(-1)[batch_index, component_index].argsort(descending=True)

    batch_index = batch_index[order]
    component_index = component_index[order]
    target_match_index = mask_iou.argmax(-1)[batch_index, component_index]

    unique_match, unique_match_index = torch.stack([batch_index, target_match_index], -1).unique(dim=0, return_inverse=True)
    unique_match_mask = torch.as_tensor([1 if index not in unique_match_index[:i] else 0 for i, index in enumerate(unique_match_index)], dtype=torch.int32)

    mismatch_index = torch.stack([row for row in torch.stack(torch.where(target_class > 0), -1) if torch.any(torch.as_tensor(row[None] != unique_match), -1).all()], 0)
    target_mismatch_class = target_class[mismatch_index[:, 0], mismatch_index[:, 1]].cpu().numpy()

    map_record = [MAPRecord(cls, conf, iou, mask) for cls, conf, iou, mask in zip(target_class[batch_index, target_match_index].cpu().numpy(), predict_conf[batch_index, component_index].cpu().numpy(), mask_iou[batch_index, component_index, target_match_index].cpu().numpy(), unique_match_mask.cpu().numpy())]

    return kin_record, roc_record, map_record, target_mismatch_class


@torch.no_grad()
def valid(test_dir, batch_size=10, pretrain: str = None):
    lhc = LhcSAM(315, 5).cuda().eval()

    if pretrain is not None and os.path.exists(pretrain):
        state = torch.load(pretrain, weights_only=True)
        lhc.load_state_dict(state["lhc"])
        print("load epoch = [%d]" % state["epoch"])

    matcher = HungarianMatcher(with_logits=True)

    l1_loss = MomentumL1Loss()
    eiou_loss = EIoULoss()
    focal_loss = FocalLoss(with_logits=True)
    mutil_class_loss = MultiClassFocalLoss(with_logits=True)
    dice_semimetric_loss = DiceSemimetricLoss((2, 3), True)

    step = np.zeros((5,))
    mean_kin = np.zeros((5,))
    mean_iou = np.zeros((5,))
    mean_conf_loss = np.zeros((1,))
    mean_mask_loss = np.zeros((1,))
    mean_class_loss = np.zeros((1,))
    mean_momentum_loss = np.zeros((1,))
    dataset = torch.utils.data.DataLoader(DatasetLoader(test_dir, n_mix=50, flat_prob=0.0), batch_size, False, drop_last=True)

    kin_records = []
    roc_records = []
    map_records = []
    mis_records = []
    # image_records = []
    for i, data in enumerate(dataset):
        image, prompt, mask_label, class_label, momentum_label = [dat.cuda() for dat in data]
        pred_conf, pred_mask, pred_class, pred_momentum = lhc([image, prompt])

        kin_record, roc_record, map_record, mis_record = prediction_process([pred_conf, pred_mask, pred_class, pred_momentum], [mask_label, class_label, momentum_label], 0.5)
        kin_records.extend(kin_record)
        roc_records.extend(roc_record)
        map_records.extend(map_record)
        mis_records.extend(mis_record)

        batch = torch.arange(batch_size)[:, None]
        indices = matcher([pred_class.transpose(1, 2), pred_mask, pred_momentum], [class_label, mask_label, momentum_label])

        pred_conf = pred_conf[batch, indices[..., 0]]
        pred_mask = torch.nn.functional.interpolate(pred_mask[batch, indices[..., 0]], mask_label.shape[-2:], mode="bilinear")
        pred_class = pred_class[batch, indices[..., 0]]
        pred_momentum = pred_momentum[batch, indices[..., 0]]

        target_mask = mask_label[batch, indices[..., 1]]
        target_class = class_label[batch, indices[..., 1]]
        target_momentum = momentum_label[batch, indices[..., 1]]

        ignore = torch.where(target_class != 0, 1.0, 0.0)

        kin = calc_momentum_iou(pred_momentum.detach(), target_momentum)
        iou = calc_mask_jaccard_metric(pred_mask.detach(), target_mask, True)
        cls = pred_class.detach().softmax(-1)[batch, torch.arange(32)[None], target_class]
        conf_loss = torch.nn.functional.l1_loss(pred_conf, (kin + iou + cls) / 3 * ignore, reduction="mean")

        dl = (dice_semimetric_loss(pred_mask, target_mask) * ignore).sum(1) / (ignore.sum(1) + 1e-3)
        fl = (focal_loss(pred_mask, target_mask) * ignore[..., None, None]).mean((2, 3)).sum(1) / (ignore.sum(1) + 1e-3)
        mask_loss = 0.25 * fl.mean() + 0.75 * dl.mean()

        # class_loss = torch.nn.functional.cross_entropy(pred_class.transpose(1, 2), target_class)
        # class_loss = mutil_class_loss(pred_class.transpose(1, 2), target_class).sum(1).mean()
        class_loss = ((mutil_class_loss(pred_class.transpose(1, 2), target_class) * ignore).sum(1) / (ignore.sum(1) + 1e-3)).mean()

        # momentum_loss = ((eiou_loss(pred_momentum, target_momentum) * ignore).sum(1) / (ignore.sum(1) + 1e-3)).mean()
        l1 = (l1_loss(pred_momentum, target_momentum) * ignore).sum(1) / (ignore.sum(1) + 1e-3)
        el = (eiou_loss(pred_momentum, target_momentum) * ignore).sum(1) / (ignore.sum(1) + 1e-3)
        momentum_loss = 0.8 * l1.mean() + 0.2 * el.mean()

        mean_conf_loss += (conf_loss.detach().cpu().numpy() - mean_conf_loss) / (i + 1)
        mean_mask_loss += (mask_loss.detach().cpu().numpy() - mean_mask_loss) / (i + 1)
        mean_class_loss += (class_loss.detach().cpu().numpy() - mean_class_loss) / (i + 1)
        mean_momentum_loss += (momentum_loss.detach().cpu().numpy() - mean_momentum_loss) / (i + 1)

        kin_tmp = np.zeros((6,))
        iou_tmp = np.zeros((6,))
        index = torch.where(target_class != 0)
        np.add.at(kin_tmp, target_class[index].cpu().numpy(), kin[index].cpu().numpy())
        np.add.at(iou_tmp, target_class[index].cpu().numpy(), iou[index].cpu().numpy())
        np.divide.at(kin_tmp, *np.unique(target_class[index].cpu().numpy(), return_counts=True))
        np.divide.at(iou_tmp, *np.unique(target_class[index].cpu().numpy(), return_counts=True))

        step_tmp = np.zeros((6,))
        np.add.at(step_tmp, *np.unique(target_class[index].cpu().numpy(), return_counts=True))
        step += np.where(step_tmp[1:] > 0, 1, 0)
        mean_kin += (np.where(step_tmp[1:] > 0, kin_tmp[1:], mean_kin) - mean_kin) / (step + 1e-3)
        mean_iou += (np.where(step_tmp[1:] > 0, iou_tmp[1:], mean_iou) - mean_iou) / (step + 1e-3)

        print(f"step: {i + 1}/{len(dataset)}, conf_loss: {mean_conf_loss[0].round(5)}, mask_loss: {mean_mask_loss[0].round(5)}, class_loss: {mean_class_loss[0].round(5)}, momentum_loss: {mean_momentum_loss[0].round(5)}      ")
        print(f"kin_iou:  [t: {mean_kin[0].round(5)}, h: {mean_kin[1].round(5)}, w/z: {mean_kin[2].round(5)}, b: {mean_kin[3].round(5)}, j: {mean_kin[4].round(5)}]      ")
        print(f"mask_iou: [t: {mean_iou[0].round(5)}, h: {mean_iou[1].round(5)}, w/z: {mean_iou[2].round(5)}, b: {mean_iou[3].round(5)}, j: {mean_iou[4].round(5)}]      ")

        # image_records.append((image.cpu().numpy(), pred_conf.cpu().numpy(), pred_class.argmax(-1).cpu().numpy(), pred_mask.cpu().numpy(), target_class.cpu().numpy(), target_mask.cpu().numpy()))
        #
        # if i + 1 == 5:
        #     break

        if i + 1 != len(dataset):
            print("\033[F" * 3, end="")

    predict = {'kin': kin_records, 'ROC': roc_records, 'mAP': [map_records, mis_records]}
    with open(f"result_{test_dir}.pkl", "wb") as file:
        pickle.dump(predict, file)

    # with open("image.pkl", "wb") as file:
    #     pickle.dump(image_records, file)


if __name__ == '__main__':
    # valid("test", 20, 'weights/state.pth')
    init()
    train(20, 120, "weights/state.pth")
