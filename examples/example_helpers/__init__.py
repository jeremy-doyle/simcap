import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import minmax_scale

sns.set_theme(style="whitegrid", palette="bright")

    
def plot_path_comparison(obs, sim, log=False):
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(16, 6))
    if log:
        obs = obs.apply(np.log)
        sim = sim.apply(np.log)
    sns.lineplot(data=obs, dashes=False, ax=ax1)
    sns.lineplot(data=sim, dashes=False, ax=ax2)
    ax2.yaxis.set_tick_params(labelbottom=True)
    ax1.set_title("Observation", fontsize=18)
    ax2.set_title("Simulation", fontsize=18)
    if log:
        title = "Cumulative Log Returns"
    else:
        title = "Cumulative Simple Returns"
    fig.suptitle(title, fontsize=24, ha="left", x=0.125, y=1.025)
    plt.show()


def plot_autocorr_comparisons(obs, sim, square=False):
    obs_log_return = obs.pct_change().dropna().apply(np.log1p)
    sim_log_return = sim.pct_change().dropna().apply(np.log1p)
    cols = obs.columns
    ax_labels = np.reshape(cols, (4, 2)).tolist()
    fig, axes = plt.subplot_mosaic(ax_labels, figsize=(16, 20))
    power = 2 if square else 1 
    for col in cols:
        data = pd.DataFrame({"obs": obs_log_return[col], "sim": sim_log_return[col]})
        #pd.plotting.autocorrelation_plot(data["sim"], axes[col])
        plot_acf(x=data["obs"] ** power, ax=axes[col], lags=50, zero=False, label="obs")
        plot_acf(x=data["sim"] ** power, ax=axes[col], lags=50, zero=False, label="sim")
        
        axes[col].set_title(col, fontsize=12)
        
        handles, labels= axes[col].get_legend_handles_labels()
        handles = [handles[1], handles[3]]
        labels = [labels[1], labels[3]]
        axes[col].legend(handles=handles, labels=labels, loc="best", numpoints=1)
    
    title_mod = "squared " if square else ""
    fig.suptitle(
        f"Autocorrelation of {title_mod.title()} Log Returns",
        fontsize=24,
        ha="left",
        x=0.125,
        y=0.925,
    )
    plt.show()
    

def return_to_volatility_correlation(obs, sim):
    obs_log_return = obs.pct_change().dropna().apply(np.log1p)
    sim_log_return = sim.pct_change().dropna().apply(np.log1p)
    
    obs_coefs = list()
    sim_coefs = list()
    
    cols = obs.columns

    for col in cols:
        obs_coef = np.corrcoef(obs_log_return[col], obs_log_return[col]**2, rowvar=False)
        sim_coef = np.corrcoef(sim_log_return[col], sim_log_return[col]**2, rowvar=False)
        obs_coefs.append(obs_coef[0,1])
        sim_coefs.append(sim_coef[0,1])
        
    data = pd.DataFrame({"obs": obs_coefs, "sim": sim_coefs}, index=cols)

    fig, ax = plt.subplots(figsize=(12, 3))

    sns.heatmap(
        minmax_scale(data).T, 
        annot=data.T, 
        xticklabels=data.index,
        yticklabels=data.columns,
        ax=ax, 
        linewidths=0.5,
        cbar=False,
        cmap="rocket",
        annot_kws={"fontsize":"large"},
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    
    fig.suptitle(
        "Correlation Between Returns and Volatility (squared log returns)",
        fontsize=24,
        ha="left",
        x=0.05,
        y=0.925,
    )
    plt.show()
    

def plot_corr_comparison(obs, sim):
    obs_corr = obs.pct_change().dropna().apply(np.log1p).corr()
    sim_corr = sim.pct_change().dropna().apply(np.log1p).corr()
    ax_labels = [["obs", "sim"]]
    fig, axes = plt.subplot_mosaic(ax_labels, figsize=(16, 6))
    sns.heatmap(obs_corr, annot=True, ax=axes["obs"], cmap="rocket")
    sns.heatmap(sim_corr, annot=True, ax=axes["sim"], cmap="rocket")
    axes["obs"].set_title("Observation", fontsize=18)
    axes["sim"].set_title("Simulation", fontsize=18)
    fig.suptitle(
        "Correlation Coefficients of 1-Period Log Returns",
        fontsize=24,
        ha="left",
        x=0.125,
        y=1.025,
    )
    plt.show()


def plot_distribution_comparisons(obs, sim, prototypes=None, asset_proto_rel=None):
    obs_log_return = obs.pct_change().dropna().apply(np.log1p)
    sim_log_return = sim.pct_change().dropna().apply(np.log1p)
    cols = obs.columns
    ax_labels = np.reshape(cols, (4, 2)).tolist()
    fig, axes = plt.subplot_mosaic(ax_labels, figsize=(16, 20))
    for col in cols:
        data = pd.DataFrame({"obs": obs_log_return[col], "sim": sim_log_return[col]})
        if prototypes is not None:
            proto = pd.Series(prototypes[asset_proto_rel[col]])
            proto = proto.pct_change().dropna().apply(np.log1p)
            data["proto"] = proto
        sns.kdeplot(data=data+1e-8, ax=axes[col], fill=True, alpha=0.25, log_scale=True)
        axes[col].set_title(col, fontsize=12)
        axes[col].set(xticklabels=[])
        sns.move_legend(axes[col], "upper left")
    fig.suptitle(
        "Distribution of 1-Period Log Returns (x-axis log scale)",
        fontsize=24,
        ha="left",
        x=0.125,
        y=0.925,
    )
    plt.show()


def calc_max_drawdown(cumulative):
    drawdown = 1 - (cumulative / cumulative.cummax())
    max_drawdown = drawdown.max()
    return -max_drawdown


def compare_statistics(obs, sim):
    obs_log_return = obs.pct_change().dropna().apply(np.log1p)
    sim_log_return = sim.pct_change().dropna().apply(np.log1p)

    ax_labels = [["min", "mean", "median"], ["max", "std", "dd"]]
    fig, axes = plt.subplot_mosaic(ax_labels, figsize=(16, 10))
    plt.subplots_adjust(wspace=0.25, hspace=0.3)

    cols = obs.columns
    lw = 0.5
    cmap = "rocket"

    # min
    data = {
        "obs": obs_log_return.min().values,
        "sim": sim_log_return.min().values
    }
    data = pd.DataFrame(data, index=cols)
    sns.heatmap(
        minmax_scale(data), 
        annot=data, 
        yticklabels=data.index,
        xticklabels=data.columns,
        ax=axes["min"], 
        linewidths=lw,
        cbar=False,
        cmap=cmap,
    )
    axes["min"].set_title("Min 1-Period Log Return", fontsize=14)

    # mean
    data = {
        "obs": obs_log_return.mean().values,
        "sim": sim_log_return.mean().values
    }
    data = pd.DataFrame(data, index=cols)
    sns.heatmap(
        minmax_scale(data), 
        annot=data, 
        yticklabels=data.index,
        xticklabels=data.columns,
        ax=axes["mean"], 
        linewidths=lw,
        cbar=False,
        cmap=cmap,
    )
    axes["mean"].set_title("Mean 1-Period Log Return", fontsize=14)

    # median
    data = {
        "obs": obs_log_return.median().values,
        "sim": sim_log_return.median().values,
    }
    data = pd.DataFrame(data, index=cols)
    sns.heatmap(
        minmax_scale(data), 
        annot=data, 
        yticklabels=data.index,
        xticklabels=data.columns,
        ax=axes["median"], 
        linewidths=lw,
        cbar=False,
        cmap=cmap,
    )
    axes["median"].set_title("Median 1-Period Log Return", fontsize=14)

    # max
    data = {
        "obs": obs_log_return.max().values,
        "sim": sim_log_return.max().values
    }
    data = pd.DataFrame(data, index=cols)
    sns.heatmap(
        minmax_scale(data), 
        annot=data, 
        yticklabels=data.index,
        xticklabels=data.columns,
        ax=axes["max"], 
        linewidths=lw,
        cbar=False,
        cmap=cmap,
    )
    axes["max"].set_title("Max 1-Period Log Return", fontsize=14)

    # std
    data = {
        "obs": obs_log_return.std().values,
        "sim": sim_log_return.std().values
    }
    data = pd.DataFrame(data, index=cols)
    sns.heatmap(
        minmax_scale(data), 
        annot=data, 
        yticklabels=data.index,
        xticklabels=data.columns,
        ax=axes["std"], 
        linewidths=lw,
        cbar=False,
        cmap=cmap,
    )
    axes["std"].set_title("1-Period Log Return StDev", fontsize=14)

    # dd
    data = {
        "obs": calc_max_drawdown(obs).values,
        "sim": calc_max_drawdown(sim).values
    }
    data = pd.DataFrame(data, index=cols)
    sns.heatmap(
        minmax_scale(data), 
        annot=data, 
        yticklabels=data.index,
        xticklabels=data.columns,
        ax=axes["dd"], 
        linewidths=lw,
        cbar=False,
        cmap=cmap,
    )
    axes["dd"].set_title("Max Cumulative Drawdown", fontsize=14)
    
    fig.suptitle("Comparative Statistics", fontsize=24, ha="left", x=0.125, y=0.975)
    plt.show()
