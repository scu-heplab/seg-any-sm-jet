import pickle
import numpy as np
from skimage import measure
from collections import namedtuple
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splprep, splev
from sklearn.metrics import confusion_matrix

MAPRecord = namedtuple("MAPRecord", ("cls", "conf", "iou", "unique"))
KINRecord = namedtuple("KINRecord", ("pred_kin", "target_kin", "unique"))
ROCRecord = namedtuple("ROCRecord", ("pred_prob", "target_cls", "unique"))


def plot_matrix(match_data: list[ROCRecord], labels):
    fig, ax = plt.subplots(figsize=(6, 6))

    match_data = ROCRecord(*zip(*match_data))
    pred_probs = np.array(match_data.pred_prob)
    target_class = np.array(match_data.target_cls)

    pred_class = np.argmax(pred_probs, -1)

    matrix = confusion_matrix(target_class, pred_class, labels=np.arange(1, 6, dtype=np.int64))
    matrix = matrix / matrix.sum(-1, keepdims=True)

    ax.set(xticks=np.arange(matrix.shape[1]), yticks=np.arange(matrix.shape[0]), xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Prediction', fontsize=12)
    ax.set_ylabel('Truth', fontsize=12)

    fmt_normalized = '.3f'
    ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1, aspect='auto')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            cell_text = f"{matrix[i, j]:{fmt_normalized}}"
            ax.text(j, i, cell_text, ha="center", va="center", color="white" if matrix[i, j] > 0.5 else "black", fontsize=9)
    fig.savefig("cm.pdf", dpi=600, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_roc(match_data: list[ROCRecord], signal_class, signal_label, background_class, background_label, step=0.01):
    fig, ax = plt.subplots(figsize=(6, 6))
    threshold = np.arange(0, 1 + step, step)

    match_data = ROCRecord(*zip(*match_data))
    unique = np.array(match_data.unique)
    pred_probs = np.array(match_data.pred_prob)
    target_class = np.array(match_data.target_cls)

    pred_probs = pred_probs[unique == 1]
    target_class = target_class[unique == 1]

    signal_mask = target_class == signal_class

    for bg_cls, bg_lab in zip(background_class, background_label):
        background_mask = target_class == bg_cls

        signal_probs = np.concatenate([pred_probs[signal_mask, signal_class, None], pred_probs[signal_mask, bg_cls, None]], -1)
        background_probs = np.concatenate([pred_probs[background_mask, signal_class, None], pred_probs[background_mask, bg_cls, None]], -1)

        signal_probs = signal_probs[np.max(signal_probs, -1) > 1 / 6]
        background_probs = background_probs[np.max(background_probs, -1) > 1 / 6]

        signal_count = np.sum(signal_probs[..., :1] > threshold[None], 0)
        background_count = np.sum(background_probs[..., :1] > threshold[None], 0)

        signal_efficiency = signal_count / signal_count[0]
        background_rejection = background_count[0] / background_count

        ax.plot(signal_efficiency, background_rejection, label=f"${bg_lab}$")

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 1e5)
    ax.set_yscale("log")
    ax.set_title(f"Signal: ${signal_label}$")

    ax.minorticks_on()
    ax.grid(which="major", linewidth=1, linestyle="--")
    ax.grid(which="minor", linewidth=0.5, linestyle="--")

    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlabel(r"Signal Efficiency $\epsilon_S$", fontsize=12)
    ax.set_ylabel(r"Background Rejection $\frac{1}{\epsilon_B}$", fontsize=12)
    ax.legend(fontsize=12)

    fig.savefig(f"roc_{signal_label.replace('/', '')}.pdf", dpi=600, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_dist(match_data: list[KINRecord], bins, ratios, draw_type, resolution=1.0):
    match_data = KINRecord(*zip(*match_data))

    unique = np.array(match_data.unique)
    pred_kin = np.array(match_data.pred_kin)
    target_kin = np.array(match_data.target_kin)

    pred_kin = pred_kin[unique == 1]
    target_kin = target_kin[unique == 1]

    pred_extra = np.where(target_kin[:, 2] - pred_kin[:, 2] > np.pi, 2 * np.pi, 0.0)
    target_extra = np.where(pred_kin[:, 2] - target_kin[:, 2] > np.pi, 2 * np.pi, 0.0)
    pred_kin[:, 2] = pred_kin[:, 2] + pred_extra
    target_kin[:, 2] = target_kin[:, 2] + target_extra

    relative = (pred_kin - target_kin) / (target_kin + 1e-8)
    relative = relative[np.where(np.logical_and(~np.any(np.isinf(relative), -1), np.logical_and(target_kin[:, 0] > 50, target_kin[:, 3] > 1)))]

    assert draw_type in ["pt/m", "eta/phi"]

    # 0：Pt，1：eta，2：phi，3：mass
    relative = relative[:, [0, 3] if draw_type == "pt/m" else [1, 2]]

    x_range = np.percentile(relative[:, 0], [0.5, 99.5])
    y_range = np.percentile(relative[:, 1], [0.5, 94 if draw_type == "pt/m" else 99.5])

    hist = np.histogram2d(relative[:, 0], relative[:, 1], bins, (x_range, y_range))[0] / relative.shape[0]
    hist = np.reshape(hist, (-1,))
    hist[np.argsort(hist)[::-1]] = np.cumsum(hist[np.argsort(hist)[::-1]])
    hist = np.reshape(hist, (bins, bins))

    def smooth_contour(contour, sigma=1.0, n_interp=100):
        smoothed = gaussian_filter(contour, sigma=[sigma, 0], mode='wrap')
        tck, u = splprep(smoothed.T, u=None, s=0.0, per=1)[:2]
        u_new = np.linspace(u.min(), u.max(), n_interp)
        x_new, y_new = splev(u_new, tck, der=0)
        return np.column_stack((x_new, y_new))

    plt.clf()
    plt.subplot(2, 2, 3)
    print(f"RMSE --- {draw_type}")
    x_limit, y_limit = (np.inf, -np.inf), (np.inf, -np.inf)
    for ratio, alpha in zip(ratios[::-1], [0.4, 0.6, 0.8]):
        region = []
        for ct in measure.find_contours(hist, ratio):
            x, y = smooth_contour(ct, 0.1, 100).T
            x = np.interp(x, (0, bins - 1), x_range)
            y = np.interp(y, (0, bins - 1), y_range)
            x_limit = min(min(x), x_limit[0]), max(max(x), x_limit[1])
            y_limit = min(min(y), y_limit[0]), max(max(y), y_limit[1])
            plt.fill(x, y, color="gray", alpha=alpha)
            region.append(np.stack([x, y], -1))
        region = np.concatenate(region, 0)
        print(f"contour:{ratio:.3f} - [{(region ** 2).sum(-1).mean() ** 0.5: .5f}]")
    if draw_type == "pt/m":
        plt.xlabel(rf"$\Delta p_T/p_T$ [$\times {resolution}$]", fontdict={"size": 12})
        plt.ylabel(rf"$\Delta m/m$ [$\times {resolution}$]", fontdict={"size": 12})
    else:
        plt.xlabel(rf"$\Delta \eta/\eta$ [$\times {resolution}$]", fontdict={"size": 12})
        plt.ylabel(rf"$\Delta \phi/\phi$ [$\times {resolution}$]", fontdict={"size": 12})
    plt.xlim(*x_limit)
    plt.ylim(*y_limit)

    x_ticks = np.arange(x_limit[0], x_limit[1] + (x_limit[1] - x_limit[0]) / bins, (x_limit[1] - x_limit[0]) / bins)[::bins // 4]
    y_ticks = np.arange(y_limit[0], y_limit[1] + (y_limit[1] - y_limit[0]) / bins, (y_limit[1] - y_limit[0]) / bins)[::bins // 4]
    plt.xticks(x_ticks, [f"{x / resolution:.2f}" for x in x_ticks])
    plt.yticks(y_ticks, [f"{y / resolution:.2f}" for y in y_ticks])

    plt.subplot(2, 2, 1)
    plt.xticks([])
    plt.xlim(*x_limit)
    plt.yticks([(i + 1) * 0.01 for i in range(5)], [f"{i + 1}%" for i in range(5)])
    plt.hist(relative[:, 0], bins, range=x_limit, weights=np.ones((relative.shape[0],)) / relative.shape[0] * hist.max())

    plt.subplot(2, 2, 4)
    plt.yticks([])
    plt.ylim(*y_limit)
    plt.xticks([(i + 1) * 0.01 for i in range(5)], [f"{i + 1}%" for i in range(5)])
    plt.hist(relative[:, 1], bins, range=y_limit, weights=np.ones((relative.shape[0],)) / relative.shape[0] * hist.max(), orientation="horizontal")

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(f"dist_{draw_type.replace('/', '')}.pdf", bbox_inches="tight", dpi=600)


def plot_pr(match_data: list[MAPRecord], iou_threshold=0.5):
    mis_data = np.array(match_data[1])
    match_data = MAPRecord(*zip(*match_data[0]))

    iou = np.array(match_data.iou)
    conf = np.array(match_data.conf)
    unique = np.array(match_data.unique)
    category = np.array(match_data.cls)

    ap = []
    label = {1: "$t$", 2: "$H$", 3: "$W/Z$", 4: "$b$", 5: "$j$"}
    fig, ax = plt.subplots(figsize=(6, 6))
    for cls in range(1, 6):
        cls_mask = category == cls
        cls_sorted_iou = iou[cls_mask][np.argsort(conf[cls_mask])[::-1]]
        cls_sorted_unique = unique[cls_mask][np.argsort(conf[cls_mask])[::-1]]

        cum_tp = np.cumsum(np.logical_and(cls_sorted_iou > iou_threshold, cls_sorted_unique == 1))
        cum_fp = np.cumsum(np.logical_or(cls_sorted_iou <= iou_threshold, cls_sorted_unique == 0))
        cls_gt = mis_data[mis_data == cls].shape[0] + cls_sorted_unique.sum()

        recall = cum_tp / cls_gt
        precision = cum_tp / (cum_tp + cum_fp)

        ap.append(np.sum((precision[1:] + precision[:-1]) * (recall[1:] - recall[:-1]) * 0.5))

        # precision = np.convolve(precision, np.ones((17,)) / 17, mode="same")

        ax.plot(recall, precision, label=f"{label[cls]}")
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0.5, 1)
    ax.legend(fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    plt.savefig(f"pr.pdf", bbox_inches="tight", dpi=600)
    print(f"AP:{np.round(ap, 3)}")
    print(f"mAP:{np.mean(ap)}")


def main():
    test_dir = "hp_wph1_500"
    with open(f"result_{test_dir}.pkl", "rb") as file:
        predict = pickle.load(file)
    plot_pr(predict["mAP"], 0.5)
    plot_matrix(predict["ROC"], ["t", "H", "W/Z", "b", "j"])
    plot_dist(predict["kin"], 80, [0.2, 0.4, 0.6], "pt/m", 1.0)
    plot_dist(predict["kin"], 80, [0.2, 0.4, 0.6], "eta/phi", 0.02)
    plot_roc(predict["ROC"], 1, "t", [2, 3, 4, 5], ["H", "W/Z", "b", "j"], 0.00001)
    plot_roc(predict["ROC"], 2, "H", [1, 3, 4, 5], ["t", "W/Z", "b", "j"], 0.00001)
    plot_roc(predict["ROC"], 4, "b", [1, 2, 3, 5], ["t", "H", "W/Z", "j"], 0.00001)
    plot_roc(predict["ROC"], 3, "W/Z", [1, 2, 4, 5], ["t", "H", "b", "j"], 0.00001)
    # print(predict)


if __name__ == "__main__":
    main()
