<p align="center" style="border: 1px solid black;">
  <img src="img/SimCAP.png" alt="SimCAP" style="width: 150px;">
</p>

# SimCAP: Simulation of Correlated Asset Prices
<img src="img/intro.png" alt="SimCAP" style="max-width: 100%;">

SimCAP was created with the intent to make it simple to generate useful simulations of correlated multivariate financial time series. SimCAP is written in Python3 and leans heavily on the excellent <a href="https://hmmlearn.readthedocs.io/en/latest/" target="_blank">hmmlearn</a> package to train the Hidden Markov Models at the core of SimCAP simulations. 

With SimCAP, just provide a pandas DataFrame of historical stock prices (or any other financial instrument) and you can easily generate thousands of simulations in minutes. Simulations resemble the original time series in both the correlations between assets and the statistical properties of the returns distributions.

Possible uses for SimCAP could be trading strategy development, portfolio optimization, portfolio risk management, financial planning, augementation of data sets for machine learning, etc.
<br><br>

## Contents
- [Installation](#installation)  
- [Quick Start](#quick-start)  
  - [Importing SimCAP and Example Data](#importing)  
  - [Generating Simulations](#generating-simulations)  
  - [Comparing Observation to Simulations](#comparison)
  - [Additional Examples](#additional-examples)
- [Stylized Facts of Financial Time Series](#stylized-facts)
  - [Returns do not follow a normal distribution and are leptokurtic.](#not-normal)  
  - [The autocorrelation of returns is insignificant.](#no-autocorrelation)  
  - [Returns are subject to volatility clustering.](#volatility-clustering)  
  - [Returns are negatively correlated with volatility.](#returns-volatility-correlation)
- [API](#api)  
- [About the Developer](#about-the-developer)

# Installation <a name="installation"></a>
SimCAP requires:
- Python >= 3.7
- NumPy >= 1.17
- SciPy >= 1.3
- scikit-learn >= 0.18
- statsmodels >= 0.13
- hmmlearn >= 0.2.7
- pandas >= 1.0

You can install SimCAP directly with pip:  
```python 
pip install simcap
```
Alternatively, you can install SimCAP directly from source:  
```python
pip install git+https://github.com/jeremy-doyle/SimCAP.git
```

# Quick Start <a name="quick-start"></a>
Daily closing prices (adjusted for dividends and splits) for the following Exchange Traded Funds (ETFs) over a 10-year period are included with the package:
<br><br>

Ticker | Description
-------|------------
SPY|SPDR S&P 500 ETF Trust
VO|Vanguard Mid-Cap Index Fund ETF
IWM|iShares Russell 2000 ETF
EFA|iShares MSCI EAFE ETF
VWO|Vanguard Emerging Markets Stock Index Fund ETF
AGG|iShares Core US Aggregate Bond ETF
IBND|SPDR Bloomberg International Corporate Bond ETF
VNQ|Vanguard Real Estate Index Fund ETF

<br> 

## Importing SimCAP and Example Data <a name="importing"></a>

```python
>>> from simcap import SimCAP
>>> from simcap.datasets import load_example_asset_price_history

>>> obs = load_example_asset_price_history()

>>> obs.head().round(2)
      SPY     VO    IWM    EFA    VWO    AGG   IBND    VNQ
0  106.80  64.47  66.44  37.22  30.88  86.69  29.32  39.60
1  107.98  65.52  67.62  37.85  31.64  86.58  29.56  39.88
2  108.55  66.00  68.04  38.31  31.94  86.44  29.68  40.04
3  108.95  65.80  68.08  38.51  31.97  86.22  29.68  40.28
4  108.67  65.91  68.00  38.66  32.14  86.15  29.95  40.48
```


## Generating Simulations <a name="generating-simulations"></a>

1. Initialize a `SimCAP`instance with the observed data
2. Call the `generate_simulations` method

For this example, parameters were chosen to:

* generate five simulations
* set the beginning values of each asset in each simulation to 1 (`begin_values="norm"`)
* output the simulations as a list of pandas dataframes
* speed things up by setting the number of parallel jobs to 3

```python
>>> model = SimCAP(obs)
>>> sims = model.generate_simulations(
>>>     n_sims=5, 
>>>     begin_values="norm",
>>>     output_as_dfs=True,
>>>     n_jobs=3,
>>> )
markov model search (n_jobs=3): 100%|██████████| 60/60 [01:27<00:00,  1.45s/it]
generating sims (n_jobs=3): 100%|██████████| 5/5 [00:00<00:00, 2465.79it/s]

>>> sims[0].head().round(4)
      SPY      VO     IWM     EFA     VWO     AGG    IBND     VNQ
0  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000
1  1.0107  1.0123  1.0194  1.0124  1.0192  1.0002  0.9966  1.0159
2  1.0138  1.0203  1.0254  1.0160  1.0379  0.9961  0.9931  1.0049
3  1.0049  1.0080  1.0082  1.0029  1.0260  0.9958  0.9879  0.9918
4  1.0061  1.0106  1.0091  1.0002  1.0166  0.9966  0.9921  0.9933
```

## Comparing Observation to Simulations <a name="comparison"></a>
```python
>>> import matplotlib.pyplot as plt
>>> import seaborn as sns

>>> sns.set_theme(style="whitegrid", palette="bright")

>>> fig, axes = plt.subplots(
>>>     ncols=2, 
>>>     nrows=3, 
>>>     sharex=True,
>>>     sharey=True, 
>>>     figsize=(8, 12),
>>> )

>>> norm_obs = obs / obs.iloc[0, :]
>>> sns.lineplot(data=norm_obs, dashes=False, ax=axes[0][0])
>>> axes[0][0].set_title("Normalized Observation", fontsize=14)

>>> for i, ax in enumerate(axes.reshape(-1)[1:]):
>>>     sns.lineplot(data=sims[i], dashes=False, ax=ax, legend=False)
>>>     ax.set_title(f"Simulation {i+1}", fontsize=14)
    
>>> plt.tight_layout()
>>> plt.show();
```
<img src="img/quick-start-sims.png" alt="Plot of five SimCAP simulations against the normalized observation" style="max-width: 100%;">

```python
>>> import numpy as np

>>> sim = sims[0]

>>> obs_log_return = norm_obs.pct_change().dropna().apply(np.log1p)
>>> sim_log_return = sim.pct_change().dropna().apply(np.log1p)

>>> obs_corr = obs_log_return.corr()
>>> sim_corr = sim_log_return.corr()

>>> corr_ax_labels = [["obs"],[ "sim"]]
>>> fig, axes = plt.subplot_mosaic(corr_ax_labels, figsize=(8, 6))
>>> sns.heatmap(obs_corr, annot=True, ax=axes["obs"], cbar=False)
>>> sns.heatmap(sim_corr, annot=True, ax=axes["sim"], cbar=False)
>>> axes["obs"].set_title("Observation", fontsize=14)
>>> axes["sim"].set_title("Simulation 1", fontsize=14)
>>> plt.tight_layout(rect=[0, 0, 1, 0.95])
>>> fig.suptitle("Correlation Coefficients of 1-Day Returns", fontsize=16)
>>> plt.show();
```
<img src="img/stylized-facts-correlation.png" alt="Inter-asset correlation coefficients for both the SimCAP simulation and the observed asset returns." style="max-width: 100%;">

## Additional Examples <a name="additional-examples"></a>

For additional examples, please see the Jupyter notebooks located in the [examples](examples/) directory of this repository.
<br><br>

# Stylized Facts of Financial Time Series <a name="stylized-facts"></a>

The mechanisms by which financial time series are generated are not fully understood. There are some statistical properties, however, that are common to financial time series. These properties are called **stylized facts**.

Stylized facts are observations that have been made in so many studies on the statistical properties of financial markets that they are widely understood to be empirical truths, to which theories and models must fit. 

We'll walk through some of the most commonly documented stylized facts and see how a SimCAP simulation conforms.

## Returns do not follow a normal distribution and are leptokurtic. <a name="not-normal"></a>

Asset returns are not normally distributed and are leptokurtic (fat tails). The degree of leptokurtosis tends to increase as the frequency of price measurement increases. In other words, there is more leptokurtosis in the distribution of 5 minute returns than there is in the distribution of weekly returns.

There is no consensus for the right distribution to model financial time series, though the following distributions are commonly used in practice:

* Normal
* Laplace
* Student's T 
* Cauchy
* Lévy

None of these distributions, however, are great representations for every asset being modeled so SimCAP simulations aren't modeled on theortical distributions *at all*; rather, they are modeled using kernel density estimates for each of the modeled instruments' empirical distributions. Therefore, simulated distributions of returns closely resemble distributions seen in the "wild".

```python
>>> import pandas as pd

>>> cols = sim.columns

>>> ax_labels = np.reshape(cols, (4, 2)).tolist()
>>> fig, axes = plt.subplot_mosaic(ax_labels, figsize=(8, 12))

>>> for col in cols:
>>>     data = pd.DataFrame({"obs": obs_log_return[col], "sim": sim_log_return[col]})
>>>     sns.kdeplot(data=data+1e-8, ax=axes[col], fill=True, alpha=0.25, log_scale=True)
>>>     axes[col].set_title(col, fontsize=12)
>>>     axes[col].set(xticklabels=[])
>>>     sns.move_legend(axes[col], "upper left")

>>> plt.tight_layout(rect=[0, 0, 1, 0.95])
>>> fig.suptitle("Distribution of 1-Day Returns (x-axis log scale)", fontsize=16)
>>> plt.show();
```

<img src="img/stylized-facts-distributions.png" alt="Plot of SimCAP simulated returns distributions against observed returns distributions." style="max-width: 100%;">

## The autocorrelation of returns is insignificant. <a name="no-autocorrelation"></a>

The autocorrelation (also known as serial correlation) of log returns in financial time series is insignificant. Considering the efficient markets hypothesis &mdash; the theory that asset prices reflect all information currently available thereby making future returns unpredictable &mdash; this lack of autocorrelation is not surprising.

Autocorrelation of returns is not present in SimCAP simulations. 

```python 
>>> from statsmodels.graphics.tsaplots import plot_acf

>>> def acf_of_returns(squared=False):
>>>     fig, axes = plt.subplot_mosaic(ax_labels, figsize=(8, 12))

>>>     power = 2 if squared else 1

>>>     for col in cols:
>>>         data = pd.DataFrame({"obs": obs_log_return[col], "sim": sim_log_return[col]})
>>>         plot_acf(x=data["obs"] ** power, ax=axes[col], lags=50, zero=False, label="obs")
>>>         plot_acf(x=data["sim"] ** power, ax=axes[col], lags=50, zero=False, label="sim")
>>>         axes[col].set_title(col, fontsize=12)
>>>         axes[col].set_xlabel("Days Lag")

>>>         handles, labels= axes[col].get_legend_handles_labels()
>>>         handles = [handles[1], handles[3]]
>>>         labels = [labels[1], labels[3]]
>>>         axes[col].legend(handles=handles, labels=labels, loc="best", numpoints=1)

>>>     plt.tight_layout(rect=[0, 0, 1, 0.95])

>>>     title_mod = "squared " if squared else ""
>>>     fig.suptitle(f"Autocorrelation of {title_mod.title()}Returns", fontsize=16)
>>>     plt.savefig(f"../img/stylized-facts-autocorrelation{title_mod.strip()}.png")
  
>>> acf_of_returns()
```

<img src="img/stylized-facts-autocorrelation.png" alt="Plot of SimCAP simulated returns autocorrelations against observed returns autocorrelations." style="max-width: 100%;">


## Returns are subject to volatility clustering. <a name="volatility-clustering"></a>

In contrast to the lack of serial dependence in returns, the autocorrelation of *squared* returns is always positive, significant, and decays slowly. This is explained by the phenomenon known as **volatility clustering**. 

The volatility of returns changes over time but its magnitude &mdash; measured by squared returns &mdash; tends to persist or "cluster". In other words, there are periods where volatility is low &mdash; and stays low &mdash; while there are other periods where volatility is high &mdash; and stays high. 

Volatility clustering is difficult to model. SimCAP simulations can exhibit volatility clustering though usually not to the degree observed in the market.

```python
>>> acf_of_returns(squared=True)
```

<img src="img/stylized-facts-autocorrelationsquared.png" alt="Plot of SimCAP autocorrelation of simulated squared returns against observed autocorrelation of squared returns." style="max-width: 100%;">

## Returns are negatively correlated with volatility. <a name="returns-volatility-correlation"></a>

Most measures of volatility of an asset are negatively correlated with the returns of that asset. Therefore, falling asset prices are typically accompanied by increased volatility and vice versa. 

Here, like we did when examining the phenomenon of volatility clustering, we are measuring volatility by squaring returns. We can see that returns and volatility are negatively correlated in the SimCAP simulation at a similar degree to what we observed in the market.

```python
>>> from sklearn.preprocessing import minmax_scale

>>> obs_coefs = list()
>>> sim_coefs = list()

>>> cols = obs.columns

>>> for col in cols:
>>>     obs_coef = np.corrcoef(obs_log_return[col], obs_log_return[col]**2, rowvar=False)
>>>     sim_coef = np.corrcoef(sim_log_return[col], sim_log_return[col]**2, rowvar=False)
>>>     obs_coefs.append(obs_coef[0,1])
>>>     sim_coefs.append(sim_coef[0,1])
 
>>> data = pd.DataFrame({"obs": obs_coefs, "sim": sim_coefs}, index=cols)

>>> fig, ax = plt.subplots(figsize=(8, 2))

>>> sns.heatmap(
>>>     minmax_scale(data).T, 
>>>     annot=data.T, 
>>>     xticklabels=data.index,
>>>     yticklabels=data.columns,
>>>     ax=ax, 
>>>     linewidths=0.5,
>>>     cbar=False,
>>> )

>>> plt.tight_layout(rect=[0, 0, 1, 0.925])
>>> fig.suptitle("Correlation Between Returns and Volatility", fontsize=16)
>>> plt.show();
```

<img src="img/stylized-facts-correlation-returns-volatility.png" alt="Correlation between returns and volatilty for both a SimCAP simulation and observed asset prices." style="max-width: 100%;">

# API <a name="api"></a>

## simcap.SimCAP 

<p style="background-color:#eaecef; padding:1em;">
  <em>class</em> simcap.<strong>SimCAP</strong>(<em>asset_price_history, prototypes=None, asset_proto_rel=None, external_factors=None, suppress_warnings=False</em>)
</p>

Simulation of Correlated Asset Prices

### Parameters

<dl>
  <dt>asset_price_history : pandas DataFrame of shape (n_periods, n_assets)</dt>
  <dd>
    DataFrame of asset price history. Assets should be in columns with periods in 
    rows. Ticker symbol or some other unique label for the asset should be in the 
    first row. It is assumed the passed DataFrame is already sorted chronologically 
    with the most recent price observation in the last row. Each of the assets in  
    this DataFrame will be modeled by the simulation.
  </dd>
  <dt>prototypes : list of ndarrays or pandas Series of shape (n_periods, ), optional</dt>
  <dd>
    A list of univariate time series of shape (n_periods, ). It is assumed each of 
    the prototypes are already sorted chronologically with the most recent  
    observation in the last element of the array/Series. Prototypical time series 
    can be used to model distributions of assets. For example, if attempting to 
    model the price of a mutual fund with only two years of bull market price 
    history, one might decide to use a stock index like the S&P 500 as a prototype 
    for the mutual fund since the stock index has a longer history with examples 
    across all market regimes.
  </dd>
  <dt>asset_proto_rel : dict, optional</dt>
  <dd>
    Required if prototypes are used. A dictionary mapping assets to prototypes. For 
    example, if the asset with ticker of "SPD" maps to the first prototype, "GSUS" 
    does not have a prototype, and "XVV" maps to the second prototype, pass the 
    dictionary: {"SPD": 0, "GSUS": None, "XVV": 1}
  </dd>
  <dt>external_factors : pandas DataFrame of shape (n_periods, n_assets), optional</dt>
  <dd>
    DataFrame of factors that are used to influence the simulation of asset prices 
    but are not simulated themselves. The DataFrame should have the same number of 
    rows as the asset_price_history DataFrame. external_factor observations for a 
    particular period should be in the same row as the asset_price_history 
    observations for the same period. For example, if row 4 of the 
    asset_price_history DataFrame are prices recorded on 2021-11-15, row 4 of the 
    external_factors DataFrame should also be values that were recorded on 
    2021-11-15.
  </dd>
  <dt>suppress_warnings : bool, default=False</dt>
  <dd>
    If True, StationaryWarning and CorrelatedExogWarning warnings are suppressed.
  </dd>
</dl>

### Examples
```python
>>> from simcap.datasets import load_example_asset_price_history
>>> from simcap import SimCAP

>>> asset_prices = load_example_asset_price_history()
>>> print(asset_prices.shape)
(2517, 8)

>>> model = SimCAP(asset_prices)
>>> sims = model.generate_simulations(
>>>     n_sims=10, 
>>>     periods_per_sim=500,
>>>     output_as_dfs=False,
>>> )
>>> print(sims.shape)
(10, 500, 8)
```

### Methods

<p style="background-color:#eaecef; padding:1em;">
  <strong>generate_simulations</strong>(<em>n_sims=100, periods_per_sim=None, begin_values="norm", hmm_search_n_iter=60, hmm_search_n_fits_per_iter=10, hmm_search_params=None, output_as_dfs=True, output_as_float32=False, n_jobs=None, verbose=True</em>)
</p>

Generate simulations using the data set at object initiation.

### Parameters
<dl>
  <dt>n_sims : int, default=100</dt>
  <dd>
    The number of simulations to generate.
  </dd>
  <dt>periods_per_sim : int, optional</dt>
  <dd>
    The number of periods, or rows, to generate in each simulation. If None 
    (default), the periods per simulation will match the number of rows in the 
    asset_price_history DataFrame.
  </dd>
  <dt>begin_values : {"start", "end", "norm"}, default="norm"</dt>
  <dd>
    The values to set for the first row of each simulation. "start" will set the 
    first row of each simulation to match the first row of asset prices in the 
    asset_price_history DataFrame. "end" will set the first row of each 
    simulation to match the last row of the asset_price_history DataFrame. 
    "norm" will set the first row of each simulation to 1 for all assets.
  </dd>
  <dt>hmm_search_n_iter: int, default=60</dt>
  <dd>
    For Hidden Markov Model search, the number of parameter settings that are 
    sampled. <code>n_iter</code> trades off runtime vs quality of the solution. If 
    exhausitive search of the parameter grid would result in fewer iterations, 
    search stops when all parameter combinations have been searched.
  </dd>
  <dt>hmm_search_n_fits_per_iter: int, default=10</dt>
  <dd>
    The number of Hidden Markov Models to be randomly initialized and evaluated 
    for each combination of parameter settings.
  </dd>
  <dt>hmm_search_params: dict, default=None</dt>
  <dd>
    For Hidden Markov Model search, dictionary with parameter names (str) as 
    keys and lists of parameters to try as dictionary values. Parameter lists 
    are sampled uniformly. All of the parameters are required:
    
    * n_states: list of ints
    * cov_window_size: list of ints
    * pca_n_components: list of floats, ints, or None
    * scale_before_pca: list of bool
    * scale_after_pca: list of bool
    
    If fit_pipeline_params is None, default is:

        dict = (
            n_states = [3, 4, 5, 6, 7, 8, 9],
            cov_window_size = [13, 21, 34, 55, 89, 144],
            pca_n_components = [0.8, 0.85, 0.9, 0.95, None],
            scale_before_pca = [True, False],
            scale_after_pca = [True, False],
        )
  </dd>
  <dt>output_as_dfs : bool, default=True</dt>
  <dd>
    If True, simulations are output as an n_sims element list of DataFrames of 
    shape (periods_per_sim, n_assets). If False, simulations are ouput as an 
    ndarray of shape (n_sims, periods_per_sim, n_assets).
  </dd>
  <dt>output_as_float32 : bool, default=False</dt>
  <dd>
    If True, convert simulation array to float32. If False, simulation will be 
    output as float64.
  </dd>
  <dt>n_jobs : int, default=None</dt>
  <dd>
    Number of jobs to run in parallel when performing Hidden Markov Model search 
    as well as when generating simulations. None means 1. -1 means using all 
    processors.
  </dd>
  <dt>verbose : bool, default=True</dt>
  <dd>
    If True, progress bar written to stdout as simulations are generated.
  </dd>
</dl>

### Returns
<dl>
  <dt>
    simulations : list (of length n_sims) of pandas DataFrames of 
    shape (n_periods, n_variables) or ndarray of shape (n_simulations, 
    n_periods, n_variables)
  </dt>
  <dd>
    If <code>output_as_dfs</code> is True, simulations are output as an n_sims element 
    list of DataFrames of shape (periods_per_sim, n_assets). If False, 
    simulations are ouput as an ndarray of shape (n_sims, 
    periods_per_sim, n_assets).
  </dd>
</dl>

# About the Developer <a name="about-the-developer"></a>
SimCAP is developed by me, [Jeremy Doyle](http://www.jeremy-doyle.com). I have a degree in finance and extensive securities industry experience. My never-ending pursuit for better analysis tools led me to data science and I fell in love with it. My passion therefore lies at the intersection of financial services and data science. 

When optimizing the allocation of my own investment portfolio I wanted to stress-test allocations against multiple alternate realities &mdash; Monte Carlo simulation &mdash; but I struggled to find a package that could generate *realistic* simulations of correlated stock prices. Try doing a Google search for "simulating correlated stock prices" and you'll find a lot of talk about Geometric Brownian Motion (doesn't easily model multiple correlated assets) or simulations based on the Cholesky decomposition (can easily generate correlated simulations but the resulting return distributions are Gaussian and do not exhibit any of the volatility clustering we see in the "wild"). Since I couldn't find a suitable tool, I decided to build my own &mdash; the result was SimCAP. 

If you have any questions, requests, or suggestions, please reach out to me at <a href="mailto:hello@jeremy-doyle.com">hello@jeremy-doyle.com</a>. I hope you find SimCAP useful.