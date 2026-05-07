import os
import json
import pandas as pd
import matplotlib.pyplot as plt

runs = [
    "cifar100_miss_small",
    "cifar100_miss_medium",
    "cifar100_miss_large",
    "siglip2_proxy_anchor_low_data"
]

def get_rank(run_dir):
    config_path = os.path.join("runs", run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
            return cfg.get("miss_r", "Unknown")
    return "Unknown"

def get_label(run_dir):
    rank = get_rank(run_dir)
    if "cifar100_miss" in run_dir:
        return f"LFM-VLM-450M-MiSS-{rank}"
    else:
        # Prompt definition request: update rank to 16 for SigLipV2
        return f"SigLipV2-MiSS-16-Control"

def main():
    fig_acc, ax_acc = plt.subplots(figsize=(6, 6))
    fig_loss, ax_loss = plt.subplots(figsize=(6, 6))

    for run in runs:
        metrics_path = os.path.join("runs", run, "metrics.csv")
        if not os.path.exists(metrics_path):
            print(f"Metrics not found for {run}")
            continue
        
        df = pd.read_csv(metrics_path)
        # Limit to epoch 10
        df = df[df["epoch"] <= 10]
        if df.empty:
            continue
            
        label = get_label(run)
        
        # 1) Plot Training Loss
        if "train_loss" in df.columns:
            ax_loss.plot(df["epoch"], df["train_loss"], marker='o', label=label, linewidth=2)
            
        # 2) Plot Accuracies
        # Determine base eval method for cifar100 runs
        config_path = os.path.join("runs", run, "config.json")
        eval_method = "knn" # Fallback
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
                eval_method = cfg.get("evaluation_method", "knn").lower()
                
        # If run only logged "val_acc", use the config representation
        if "val_acc" in df.columns:
            acc_name = "KNN" if eval_method == "knn" else "Linear Probe"
            ax_acc.plot(df["epoch"], df["val_acc"], marker='^', linestyle='--', label=f"{label} ({acc_name})")
            
        # If run logs explicitly val_knn_acc / val_linear_acc (like SigLip2)
        if "val_knn_acc" in df.columns:
            ax_acc.plot(df["epoch"], df["val_knn_acc"], marker='s', linestyle='-', label=f"{label} (KNN)")
        if "val_linear_acc" in df.columns:
            ax_acc.plot(df["epoch"], df["val_linear_acc"], marker='D', linestyle=':', label=f"{label} (Linear Probe)")

    # Formatting Accuracy Axis
    ax_acc.set_xlabel("Epoch", fontsize=12)
    ax_acc.set_ylabel("Accuracy", fontsize=12)
    ax_acc.set_title("Accuracy vs Epoch", fontsize=14)
    ax_acc.legend(fontsize=9)
    ax_acc.grid(True, linestyle="--", alpha=0.7)

    # Formatting Loss Axis
    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Training Loss", fontsize=12)
    ax_loss.set_title("Training Loss vs Epoch", fontsize=14)
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, linestyle="--", alpha=0.7)

    fig_acc.tight_layout()
    fig_acc.savefig("performance_plot_acc.svg", format='svg')
    
    fig_loss.tight_layout()
    fig_loss.savefig("performance_plot_loss.svg", format='svg')
    print("Successfully saved plots to performance_plot_acc.svg and performance_plot_loss.svg")

if __name__ == "__main__":
    main()