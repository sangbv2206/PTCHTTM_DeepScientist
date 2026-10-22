import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_rl_results(out_dir="."):
    # Prefer run_1 if exists, then run_0, otherwise out_dir
    if os.path.exists(os.path.join("run_1", "final_info.json")):
        target_dir = "run_1"
    elif os.path.exists(os.path.join("run_0", "final_info.json")):
        target_dir = "run_0"
    else:
        target_dir = out_dir
    info_path = os.path.join(target_dir, "final_info.json")

    if not os.path.exists(info_path):
        print(f"[Plot RL] Info path {info_path} not found. Skipping plot generation.")
        return

    with open(info_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dynamics = data.get("training_dynamics", {})
    rewards = dynamics.get("episode_rewards", [])
    td_losses = dynamics.get("td_losses", [])
    episodes = list(range(1, len(rewards) + 1))

    # Moving average helper
    def moving_avg(arr, window=10):
        if len(arr) < window:
            return arr
        return np.convolve(arr, np.ones(window) / window, mode="valid")

    os.makedirs(out_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default")

    # -------------------------------------------------------------
    # FIGURE 1: Episodic Return & Moving Average Convergence
    # -------------------------------------------------------------
    if episodes:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
        ax.plot(episodes, rewards, color="#3498db", alpha=0.35, label="Raw Episodic Return")
        if len(rewards) >= 10:
            ma_rewards = moving_avg(rewards, 10)
            ax.plot(range(10, len(rewards) + 1), ma_rewards, color="#2980b9", linewidth=2.5, label="10-Episode Moving Average")

        ax.axhline(y=195.0, color="#e74c3c", linestyle="--", linewidth=1.5, label=r"Solved Threshold (Reward $\geq$ 195)")
        ax.set_title("Episodic Return Dynamics and Cumulative Reward Convergence", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Training Episodes", fontsize=11)
        ax.set_ylabel("Cumulative Episodic Return", fontsize=11)
        ax.set_ylim(0, max(rewards) * 1.15 if max(rewards) > 0 else 215)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#bdc3c7")
        plt.tight_layout()

        fig1_path = os.path.join(out_dir, "figure_1.png")
        plt.savefig(fig1_path, dpi=300)
        plt.close()
        print(f"[Plot RL] Saved {fig1_path}")

    # -------------------------------------------------------------
    # FIGURE 2: TD Loss Curve & Benchmark Comparison
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

    # Subplot A: Temporal Difference Loss
    if td_losses:
        loss_episodes = list(range(1, len(td_losses) + 1))
        ax1.plot(loss_episodes, td_losses, color="#e67e22", linewidth=2.0, label="TD Loss")
    ax1.set_title("Temporal Difference (TD) Loss Decay", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Training Episodes", fontsize=10)
    ax1.set_ylabel("Smooth L1 Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Subplot B: Benchmark Performance Bar Chart (Baseline vs Proposed)
    b_reward = None
    p_reward = None

    if os.path.exists(os.path.join("run_0", "final_info.json")):
        try:
            with open(os.path.join("run_0", "final_info.json"), "r", encoding="utf-8") as f:
                b_info = json.load(f)
                b_reward = float(b_info.get("mean_episode_reward", b_info.get("mean_reward", 0.0)))
        except Exception:
            pass

    if os.path.exists(os.path.join("run_1", "final_info.json")):
        try:
            with open(os.path.join("run_1", "final_info.json"), "r", encoding="utf-8") as f:
                p_info = json.load(f)
                p_reward = float(p_info.get("mean_episode_reward", p_info.get("mean_reward", 0.0)))
        except Exception:
            pass
    elif "mean_episode_reward" in data:
        p_reward = float(data.get("mean_episode_reward", 0.0))

    # Đọc nhãn hiển thị nếu có từ run_labels.json
    labels_map = {}
    if os.path.exists("run_labels.json"):
        try:
            with open("run_labels.json", "r", encoding="utf-8") as rlf:
                labels_map = json.load(rlf)
        except Exception:
            pass
    b_label = labels_map.get("run_0", "Standard Baseline (DQN)")
    p_label = labels_map.get("run_1", "Proposed Agent")

    if b_reward is not None and p_reward is not None:
        models = [b_label, p_label]
        scores = [b_reward, p_reward]
        colors = ["#4a90e2", "#2ecc71"]
    elif p_reward is not None:
        models = [p_label]
        scores = [p_reward]
        colors = ["#2ecc71"]
    else:
        models = []
        scores = []
        colors = []

    if models:
        bars = ax2.bar(models, scores, color=colors, width=0.45 if len(models) > 1 else 0.3, edgecolor="black", linewidth=0.8)
        ax2.set_title("Policy Benchmark Comparison (Mean Return)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Mean Return (Last 20 Episodes)", fontsize=10)
        ax2.set_ylim(0, max(scores) * 1.25 if max(scores) > 0 else 200)
        ax2.grid(axis="y", linestyle="--", alpha=0.5)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=0, ha="center", fontsize=9.5, fontweight="bold")

        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                         fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig2_path = os.path.join(out_dir, "figure_2.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"[Plot RL] Saved {fig2_path}")


def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=".")
    args = parser.parse_args()
    plot_rl_results(args.out_dir)


if __name__ == "__main__":
    main()
