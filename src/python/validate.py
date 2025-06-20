import os
import torch
import pickle
import argparse
import numpy as np
import torch.utils.data
from collections import namedtuple
from dataset_loader import DatasetLoader
from model import LhcSAM, MomentumL1Loss, EIoULoss, FocalLoss, HungarianMatcher, MultiClassFocalLoss, DiceSemimetricLoss, calc_bbox, calc_momentum_iou, calc_mask_jaccard_metric

MAPRecord = namedtuple('MAPRecord', ('cls', 'conf', 'iou', 'unique'))
KINRecord = namedtuple('KINRecord', ('pred_kin', 'target_kin', 'unique'))
ROCRecord = namedtuple('ROCRecord', ('pred_prob', 'target_cls', 'unique'))


@torch.no_grad()
def prediction_process(predicts, targets, conf_threshold=0.3):
    target_mask, target_class, target_momentum = targets
    predict_conf, predict_mask, predict_class, predict_momentum = predicts

    predict_mask = torch.nn.functional.interpolate(predict_mask, target_mask.shape[-2:], mode="bilinear")

    # kin_iou = calc_momentum_iou(predict_momentum[:, :, None], target_momentum[:, None])
    # kin_iou = torch.where(torch.as_tensor(torch.unsqueeze(target_class, 1) != 0), kin_iou, -torch.inf)

    mask_iou = calc_mask_jaccard_metric(predict_mask[:, :, None], target_mask[:, None], True)
    mask_iou = torch.where(torch.as_tensor(torch.unsqueeze(target_class, 1) != 0), mask_iou, -torch.inf)

    iou = mask_iou  # (kin_iou + mask_iou) / 2

    iou_ignore = torch.where(iou.amax(-1) >= 0.5, 1.0, 0.0)
    conf_ignore = torch.where(predict_conf > conf_threshold, 1.0, 0.0)

    batch_index, component_index = torch.where(iou_ignore * conf_ignore > 0)

    order = iou.amax(-1)[batch_index, component_index].argsort(descending=True)

    batch_index = batch_index[order]
    component_index = component_index[order]
    target_match_index = iou.argmax(-1)[batch_index, component_index]

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
def valid(test_dir, output_path, batch_size=10, pretrain: str = None):
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

        if i + 1 != len(dataset):
            print("\033[F" * 3, end="")

    print(f"\nSaving prediction results to: {output_path}")
    predict = {'kin': kin_records, 'ROC': roc_records, 'mAP': [map_records, mis_records]}
    with open(output_path, "wb") as file:
        pickle.dump(predict, file)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Run validation and inference on the LhcSAM model.")

    parser.add_argument('--test-dir', type=str, required=True,
                        help='Path to the directory containing the test dataset.')

    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save the validation results pkl file.')

    parser.add_argument('--batch-size', type=int, default=20,
                        help='Batch size for validation (default: 20).')

    parser.add_argument('--pretrain', type=str, default=None,
                        help='Path to the pretrained model weights file (.pth). If not provided, the model starts from scratch.')

    args = parser.parse_args()

    valid(test_dir=args.test_dir, output_path=args.output_path, batch_size=args.batch_size, pretrain=args.pretrained_weights)


if __name__ == '__main__':
    main()
