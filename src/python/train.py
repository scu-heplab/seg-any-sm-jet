import os
import torch
import numpy as np
import torch.utils.data
from dataset_loader import DatasetLoader
from torch.utils.data.distributed import DistributedSampler
from model import LhcSAM, MomentumL1Loss, EIoULoss, FocalLoss, HungarianMatcher, MultiClassFocalLoss, DiceSemimetricLoss, calc_momentum_iou, calc_mask_jaccard_metric


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
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            print("load epoch = [%d], lr = [%.6f]" % (state["epoch"], optimizer.param_groups[0]["lr"]))
            start = torch.as_tensor(state["epoch"]).cuda()
            # start = torch.as_tensor(0).cuda()
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


if __name__ == '__main__':
    init()
    train(20, 400, "weights/state.pth")
