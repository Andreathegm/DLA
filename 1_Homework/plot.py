import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def load_results(results_dir="results", classify_dir="classify"):
    """Carica tutti i .pt da results/ e classify/ e li unisce per modello."""
    data = {}
    
    for f in sorted(os.listdir(results_dir)):
        if f.endswith(".pt"):
            name = f.replace(".pt", "")
            data[name] = torch.load(os.path.join(results_dir, f))
    
    for f in sorted(os.listdir(classify_dir)):
        if f.endswith(".pt"):
            name = f.replace(".pt", "")
            if name in data:
                data[name].update(torch.load(os.path.join(classify_dir, f)))
    
    return data


def plot_map_comparison(data: dict, save_path="plots/map_comparison.png"):
    """Barchart mAP per modello."""
    os.makedirs("plots", exist_ok=True)
    
    models = list(data.keys())
    maps   = [data[m]["mAP"] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(models, maps, color="steelblue", edgecolor="black")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylabel("mAP")
    ax.set_title("Retrieval mAP per backbone")
    ax.set_ylim(0, max(maps) * 1.2)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_nmc_accuracy(data: dict, save_path="plots/nmc_accuracy.png"):
    """Barchart accuracy NMC per modello."""
    os.makedirs("plots", exist_ok=True)
    
    models = [m for m in data if "acc" in data[m]]
    accs   = [data[m]["acc"] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(models, accs, color="darkorange", edgecolor="black")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylabel("Accuracy")
    ax.set_title("NMC Accuracy per backbone")
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_per_class_accuracy(data: dict, model_name: str, save_path=None):
    """Barchart accuracy per classe per un singolo modello."""
    os.makedirs("plots", exist_ok=True)
    save_path = save_path or f"plots/{model_name}_per_class_accuracy.png"
    
    acc_per_class = data[model_name]["acc_per_class"]
    if isinstance(acc_per_class, torch.Tensor):
        acc_per_class = acc_per_class.cpu().numpy()
    
    classes = np.arange(len(acc_per_class))
    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(classes, acc_per_class, color="darkorange", edgecolor="black")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy per classe — {model_name}")
    ax.set_xticks(classes)
    ax.set_ylim(0, 1.1)
    ax.axhline(acc_per_class.mean(), color="red", linestyle="--",
               label=f"Mean = {acc_per_class.mean():.3f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_per_class_ap(data: dict, model_name: str, save_path=None):
    """Barchart AP per classe per un singolo modello."""
    os.makedirs("plots", exist_ok=True)
    save_path = save_path or f"plots/{model_name}_per_class_ap.png"
    
    ap = data[model_name]["avg_precision_per_class"]
    if isinstance(ap, torch.Tensor):
        ap = ap.cpu().numpy()
    
    classes = np.arange(len(ap))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(classes, ap, color="steelblue", edgecolor="black")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Average Precision")
    ax.set_title(f"AP per classe — {model_name}")
    ax.set_xticks(classes)
    ax.axhline(ap.mean(), color="red", linestyle="--", label=f"mAP = {ap.mean():.3f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_map_vs_accuracy(data: dict, save_path="plots/map_vs_accuracy.png"):
    """Scatter mAP vs NMC accuracy — utile per vedere correlazione."""
    os.makedirs("plots", exist_ok=True)
    
    models = [m for m in data if "mAP" in data[m] and "acc" in data[m]]
    maps   = [data[m]["mAP"] for m in models]
    accs   = [data[m]["acc"] for m in models]
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(maps, accs, color="purple", s=100, zorder=3)
    
    for m, x, y in zip(models, maps, accs):
        ax.annotate(m, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    
    ax.set_xlabel("mAP (retrieval)")
    ax.set_ylabel("Accuracy (NMC)")
    ax.set_title("mAP vs NMC Accuracy")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_all(results_dir="results", classify_dir="classify"):
    data = load_results(results_dir, classify_dir)
    
    plot_map_comparison(data)
    plot_nmc_accuracy(data)
    plot_map_vs_accuracy(data)
    
    # Per ogni modello, plotta anche l'AP per classe
    for model_name in data:
        if "avg_precision_per_class" in data[model_name]:
            plot_per_class_ap(data, model_name)
        if "acc_per_class" in data[model_name]:
            plot_per_class_accuracy(data, model_name)


plot_all()

