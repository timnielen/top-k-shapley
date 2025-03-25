import matplotlib.pyplot as plt
import math
import numpy as np
import time

def plot(results, labels, measures=["numeric", "ratio", "epsilon", "mse"], save=False, filepath=f"results/{time.time()}.pdf"):
    plt.style.use(plt.style.library['ggplot'])
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['lines.markersize'] = 10
    fig, axes = plt.subplots(1, len(measures), figsize=(len(measures)*8,5), dpi=1000)
    if type(axes) != np.ndarray:
        axes = [axes]
    axes = {measure: axes[i] for i, measure in enumerate(measures)}   

    for measure in measures:
        axes[measure].set_xlabel(labels[measure][0])
        axes[measure].set_ylabel(labels[measure][1])

    for name, x, result in results:
        for measure, (avg, se) in result.items():
            if not measure in measures:
                return
            axes[measure].plot(x, avg, ".-", label=name, linewidth=2.0)
            axes[measure].fill_between(x, (avg-se), (avg+se), alpha=.3)
    
    handles, labels = axes[measures[0]].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=4, bbox_to_anchor=(0.5, -0.15))

    if save:
        plt.savefig(filepath, bbox_inches='tight')
    plt.show()