import pickle
import numpy as np
from skimage import measure
from collections import namedtuple
from scipy.stats import gaussian_kde
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
    relative = relative[np.where(np.logical_and(~np.any(np.isinf(relative), -1), np.logical_and(target_kin[:, 0] > 200, target_kin[:, 3] > 1)))]

    assert draw_type in ["pt/m", "eta/phi"]

    relative = relative[:, [0, 3] if draw_type == "pt/m" else [1, 2]]

    x_range_percentile = np.percentile(relative[:, 0], [0.5, 99.5])
    y_range_percentile = np.percentile(relative[:, 1], [0.5, 94 if draw_type == "pt/m" else 99.5])

    hist = np.histogram2d(relative[:, 0], relative[:, 1], bins, (x_range_percentile, y_range_percentile))[0]

    if hist.sum() == 0:
        print(f"Warning: No data found in the specified range for {draw_type}. Skipping plot.")
        return

    hist = hist / relative.shape[0]
    hist = np.reshape(hist, (-1,))
    hist[np.argsort(hist)[::-1]] = np.cumsum(hist[np.argsort(hist)[::-1]])
    hist = np.reshape(hist, (bins, bins))

    def smooth_contour(contour, sigma=1.0, n_interp=100):
        if contour.shape[0] < 4: return contour
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
            x = np.interp(x, (0, bins - 1), x_range_percentile)
            y = np.interp(y, (0, bins - 1), y_range_percentile)
            x_limit = min(min(x), x_limit[0]), max(max(x), x_limit[1])
            y_limit = min(min(y), y_limit[0]), max(max(y), y_limit[1])
            plt.fill(x, y, color="gray", alpha=alpha)
            region.append(np.stack([x, y], -1))
        if not region: continue
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
    plt.xticks(x_ticks, [f"{x / resolution:.2f}" for x in x_ticks], fontsize=12)
    plt.yticks(y_ticks, [f"{y / resolution:.2f}" for y in y_ticks], fontsize=12)

    mask_in_2d_range = (
            (relative[:, 0] >= x_range_percentile[0]) & (relative[:, 0] <= x_range_percentile[1]) &
            (relative[:, 1] >= y_range_percentile[0]) & (relative[:, 1] <= y_range_percentile[1])
    )
    relative_in_range = relative[mask_in_2d_range]

    plt.subplot(2, 2, 1)
    if relative_in_range.shape[0] > 1:
        kde_x = gaussian_kde(relative_in_range[:, 0])
        x_grid = np.linspace(x_limit[0], x_limit[1], 400)
        pdf_x = kde_x(x_grid)
        plt.fill_between(x_grid, pdf_x, color="gray", alpha=0.5)

    plt.subplot(2, 2, 4)
    if relative_in_range.shape[0] > 1:
        kde_y = gaussian_kde(relative_in_range[:, 1])
        y_grid = np.linspace(y_limit[0], y_limit[1], 400)
        pdf_y = kde_y(y_grid)
        plt.fill_betweenx(y_grid, pdf_y, color="gray", alpha=0.5)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for label, mass_range, color in zip(["t", "H", "W/Z", "b"], [(160, 180), (120, 130), (70, 100), (3, 5)], ["#507936", "#B32142", "#EDA01F", "#346C9C"]):
        particle_mask = np.logical_and(mass_range[0] < target_kin[:, 3], target_kin[:, 3] < mass_range[1])
        if not np.any(particle_mask): continue

        fi_pred_kin = pred_kin[particle_mask]
        fi_target_kin = target_kin[particle_mask]

        relative_fi = (fi_pred_kin - fi_target_kin) / (fi_target_kin + 1e-8)
        if relative_fi.shape[0] < 2: continue

        inf_mask = np.logical_and(~np.any(np.isinf(relative_fi), -1), fi_target_kin[:, 0] > 200)
        relative_fi = relative_fi[inf_mask]
        if relative_fi.shape[0] < 2: continue

        relative_fi = relative_fi[:, [0, 3] if draw_type == "pt/m" else [1, 2]]

        x_range_fi = np.percentile(relative_fi[:, 0], [0.5, 99.5])
        y_range_fi = np.percentile(relative_fi[:, 1], [0.5, 94 if draw_type == "pt/m" else 99.5])

        hist_fi, _, _ = np.histogram2d(relative_fi[:, 0], relative_fi[:, 1], bins, (x_range_fi, y_range_fi))
        if hist_fi.sum() == 0: continue

        hist_fi_norm = hist_fi / relative_fi.shape[0]
        hist_fi = np.reshape(hist_fi_norm, (-1,))
        hist_fi[np.argsort(hist_fi)[::-1]] = np.cumsum(hist_fi[np.argsort(hist_fi)[::-1]])
        hist_fi = np.reshape(hist_fi, (bins, bins))

        plt.subplot(2, 2, 3)
        print(f"{label}-RMSE --- {draw_type}")
        region = []
        for i, ct in enumerate(measure.find_contours(hist_fi, ratios[1])):
            x, y = smooth_contour(ct, 0.1, 100).T
            x = np.interp(x, (0, bins - 1), x_range_fi)
            y = np.interp(y, (0, bins - 1), y_range_fi)
            line_label = label if i == 0 else None
            plt.plot(x, y, color=color, lw=2, label=line_label)
            region.append(np.stack([x, y], -1))

        if not region: continue
        region = np.concatenate(region, 0)
        print(f"contour:{ratios[1]:.3f} - [{(region ** 2).sum(-1).mean() ** 0.5: .5f}]")

        mask_fi_in_range = (
                (relative_fi[:, 0] >= x_range_percentile[0]) & (relative_fi[:, 0] <= x_range_percentile[1]) &
                (relative_fi[:, 1] >= y_range_percentile[0]) & (relative_fi[:, 1] <= y_range_percentile[1])
        )
        relative_fi_in_range = relative_fi[mask_fi_in_range]
        if relative_fi_in_range.shape[0] < 2: continue

        plt.subplot(2, 2, 1)
        kde_fi_x = gaussian_kde(relative_fi_in_range[:, 0])
        x_grid = np.linspace(x_limit[0], x_limit[1], 400)
        pdf_fi_x = kde_fi_x(x_grid)
        plt.plot(x_grid, pdf_fi_x, color=color)

        plt.subplot(2, 2, 4)
        kde_fi_y = gaussian_kde(relative_fi_in_range[:, 1])
        y_grid = np.linspace(y_limit[0], y_limit[1], 400)
        pdf_fi_y = kde_fi_y(y_grid)
        plt.plot(pdf_fi_y, y_grid, color=color)

    plt.subplot(2, 2, 3)
    handles, labels = plt.gca().get_legend_handles_labels()

    plt.subplot(2, 2, 2)
    plt.axis("off")
    if handles:
        plt.legend(handles, labels, loc="center", fontsize=12)

    plt.subplot(2, 2, 1)
    plt.ylim(0)
    plt.xticks([])
    plt.xlim(*x_limit)
    plt.yticks(plt.gca().get_yticks()[1:-1], fontsize=12)
    plt.ylabel("PDF", fontsize=12)
    plt.subplot(2, 2, 4)
    plt.xlim(0)
    plt.yticks([])
    plt.ylim(*y_limit)
    plt.xticks(plt.gca().get_xticks()[1:-1], fontsize=12)
    plt.xlabel("PDF", fontsize=12)

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
    color = {1: "#507936", 2: "#B32142", 3: "#EDA01F", 4: "#346C9C", 5: "#888888"}
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

        ax.plot(recall, precision, label=f"{label[cls]}", color=f"{color[cls]}", linewidth=2.0)
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
    test_dir = "test_p200"
    with open(f"result_{test_dir}.pkl", "rb") as file:
        predict = pickle.load(file)
    plot_pr(predict["mAP"], 0.5)
    plot_matrix(predict["ROC"], ["t", "H", "W/Z", "b", "j"])
    plot_dist(predict["kin"], 100, [0.2, 0.4, 0.6], "pt/m", 1.0)
    plot_dist(predict["kin"], 100, [0.2, 0.4, 0.6], "eta/phi", 0.02)
    # plot_roc(predict["ROC"], 1, "t", [2, 3, 4, 5], ["H", "W/Z", "b", "j"], 0.00001)
    # plot_roc(predict["ROC"], 2, "H", [1, 3, 4, 5], ["t", "W/Z", "b", "j"], 0.00001)
    # plot_roc(predict["ROC"], 4, "b", [1, 2, 3, 5], ["t", "H", "W/Z", "j"], 0.00001)
    # plot_roc(predict["ROC"], 3, "W/Z", [1, 2, 4, 5], ["t", "H", "b", "j"], 0.00001)
    # print(predict)


if __name__ == "__main__":
    main()
