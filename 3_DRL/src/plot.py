import matplotlib.pyplot as plt
import os
from IPython.display import display

def plot_results(*triples, save_dir="plots", plot_grid=False, show=False, save=True):
    os.makedirs(save_dir, exist_ok=True)

    for i, (x, y, xlabel, ylabel) in enumerate(triples):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(plot_grid)

        if save:
            filename = f"{xlabel}_{ylabel}.png".replace(" ", "_")
            filepath = os.path.join(save_dir, filename)
            fig.savefig(filepath, dpi=200, bbox_inches="tight")

        if show:
            display(fig)

        plt.close(fig)