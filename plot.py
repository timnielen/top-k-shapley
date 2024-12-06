import matplotlib.pyplot as plt
import math
import numpy as np
import time

def plot(results, step_interval, types = ["topk", "mse"], metric="ratio", save=False, filepath=f"results/{time.time()}.pdf", start_at_x=0):
    plt.style.use(plt.style.library['ggplot'])
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['lines.markersize'] = 10
    fig, axes = plt.subplots(1, len(types), figsize=(len(types)*8,5), dpi=1000)
    if type(axes) != np.ndarray:
        axes = [axes]
    axes = {types[i]: axes[i] for i in range(len(types))}   

    for name, (topk, topk_SE, mse, mse_SE, percentage, percentage_SE) in results:
        start_index = max(math.floor(start_at_x / step_interval)-1, 0)
        topk, topk_SE, mse, mse_SE, percentage, percentage_SE = topk[start_index:], topk_SE[start_index:], mse[start_index:], mse_SE[start_index:], percentage[start_index:], percentage_SE[start_index:]
        x = (np.arange(topk.shape[0])+1+start_index)*step_interval
        if "topk" in types:
            axes["topk"].plot(x, topk, ".-", label=name, linewidth=2.0)
            axes["topk"].fill_between(x, (topk-topk_SE), (topk+topk_SE), alpha=.3)
        if "percentage" in types:
            axes["percentage"].plot(x, percentage, ".-", label=name, linewidth=2.0)
            # axes["percentage"].fill_between(x, (percentage-percentage_SE), (percentage+percentage_SE), alpha=.3)
        if "mse" in types:
            axes["mse"].plot(x, mse, ".-", label=name, linewidth=2.0)
            axes["mse"].fill_between(x, (mse-mse_SE), (mse+mse_SE), alpha=.3)
    
    handles, labels = axes[types[0]].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=4, bbox_to_anchor=(0.5, -0.15))
    
    if "topk" in types:
        axes["topk"].set_xlabel("T")
        if metric == "ratio":
            axes["topk"].set_ylabel("φ")
        else:
            axes["topk"].set_ylabel("ψ")
    if "mse" in types:
        axes["mse"].set_xlabel("T")
        axes["mse"].set_ylabel("MSE")

    if save:
        plt.savefig(filepath, bbox_inches='tight')
    plt.show()
    
def plot_order(results, step_interval, types = ["binary", "kendall", "spearman"], save=False, filepath=f"results/order_{time.time()}.pdf", start_at_x=0):
    plt.style.use(plt.style.library['ggplot'])
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['lines.markersize'] = 10
    fig, axes = plt.subplots(1, len(types), figsize=(len(types)*8,5), dpi=1000)
    if type(axes) != np.ndarray:
        axes = [axes]
    axes = {types[i]: axes[i] for i in range(len(types))}   

    for name, (avg_binary, avg_kendall, avg_spearman) in results:
        start_index = max(math.floor(start_at_x / step_interval)-1, 0)
        avg_binary, avg_kendall, avg_spearman = avg_binary[start_index:], avg_kendall[start_index:], avg_spearman[start_index:]
        x = (np.arange(avg_binary.shape[0])+1+start_index)*step_interval
        if "binary" in types:
            axes["binary"].plot(x, avg_binary, ".-", label=name, linewidth=2.0)
        if "kendall" in types:
            axes["kendall"].plot(x, avg_kendall, ".-", label=name, linewidth=2.0)
        if "spearman" in types:
            axes["spearman"].plot(x, avg_spearman, ".-", label=name, linewidth=2.0)
    
    handles, labels = axes[types[0]].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=4, bbox_to_anchor=(0.5, -0.15))
    
    for ax in axes:
        axes[ax].set_xlabel("T")

    if save:
        plt.savefig(filepath, bbox_inches='tight')
    plt.show()
    
def plot_fixedBudget(results, K, save=False, filepath="fixedK.pdf"):
    plt.style.use(plt.style.library['ggplot'])
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['lines.markersize'] = 10
    fig, axis = plt.subplots(1, 1, figsize=(8,5), dpi=1000)
    
    for name, (precision, precision_SE) in results:
        axis.plot(K, precision, ".-", label=name, linewidth=2.0)
        axis.fill_between(K, (precision-precision_SE), (precision+precision_SE), alpha=.3)
        
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=4, bbox_to_anchor=(0.5, -0.15))
    
    axis.set_xlabel("k")
    axis.set_ylabel("φ")
    
    if save:
        plt.savefig(filepath, bbox_inches='tight')
    plt.show()