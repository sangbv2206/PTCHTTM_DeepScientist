import json
import os
import os.path as osp
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# LOAD FINAL RESULTS:
datasets = ["circle", "dino", "line", "moons"]
panel_tags = ["(a)", "(b)", "(c)", "(d)"]
folders = sorted(os.listdir("./"))
final_results = {}
train_info = {}


def smooth(x, window_len=10, window='hanning'):
    x = np.asarray(x)
    if len(x) < window_len + 1:
        return x
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        info_p = osp.join(folder, "final_info.json")
        res_p = osp.join(folder, "all_results.pkl")
        if not (osp.exists(info_p) and osp.exists(res_p)):
            continue
        with open(info_p, "r") as f:
            final_results[folder] = json.load(f)
        with open(res_p, "rb") as f:
            train_info[folder] = pickle.load(f)

# LEGEND LABELS: đọc tên hiển thị từ run_labels.json (do pipeline ghi ra) thay vì tên thư mục nội bộ run_0/run_1
labels = {"run_0": "Baseline DDPM", "run_1": "Proposed"}
if osp.exists("run_labels.json"):
    try:
        with open("run_labels.json", "r", encoding="utf-8") as f:
            labels.update(json.load(f))
    except Exception:
        pass

runs = list(final_results.keys())
for run in runs:
    if run not in labels:
        labels[run] = run.replace("_", " ").title()

# Bảng màu cố định, phân biệt rõ baseline / proposed
base_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
colors = [base_colors[i % len(base_colors)] for i in range(len(runs))]

# Plot 1: Training loss per dataset (2x2 panels)
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for j, dataset in enumerate(datasets):
    ax = axs[j // 2, j % 2]
    for i, run in enumerate(runs):
        if dataset not in train_info[run]:
            continue
        curve = smooth(train_info[run][dataset]["train_losses"], window_len=25)
        ax.plot(curve, label=labels[run], color=colors[i], lw=1.6)
    ax.set_title(f"{panel_tags[j]} {dataset.title()}")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training objective")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
plt.tight_layout()
plt.savefig("train_loss.png", dpi=300)
plt.close()

# Plot 2: Generated samples — one row per method, one column per dataset
num_runs = len(runs)
if num_runs > 0:
    fig, axs = plt.subplots(num_runs, 4, figsize=(14, 3.6 * num_runs), squeeze=False)
    for i, run in enumerate(runs):
        for j, dataset in enumerate(datasets):
            ax = axs[i, j]
            if dataset in train_info[run]:
                images = train_info[run][dataset]["images"]
                ax.scatter(images[:, 0], images[:, 1], s=1, alpha=0.25, color=colors[i])
            tag = panel_tags[j] if i == 0 else ""
            ax.set_title(f"{tag} {dataset.title()}".strip())
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks([])
            ax.set_yticks([])
        axs[i, 0].set_ylabel(labels[run], fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("generated_images.png", dpi=300)
    plt.close()

# Plot 3: KL divergence comparison across datasets
if num_runs >= 1:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(datasets))
    width = 0.8 / max(num_runs, 1)
    for i, run in enumerate(runs):
        kl_vals = [final_results.get(run, {}).get(d, {}).get("means", {}).get("kl_divergence", 0.0) for d in datasets]
        offset = (i - (num_runs - 1) / 2) * width
        rects = ax.bar(x + offset, kl_vals, width, label=labels[run], color=colors[i],
                       alpha=0.9, edgecolor="black", linewidth=0.8)
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Estimated KL divergence (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in datasets])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("kl_divergence_comparison.png", dpi=300)
    plt.close()
