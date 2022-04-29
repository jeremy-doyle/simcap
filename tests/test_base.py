# -*- coding: utf-8 -*-

import pytest
from simcap import BaseSimulation as bs
from simcap.datasets import load_example_asset_price_history
import numpy as np
import pandas as pd
from scipy.stats import t, chi2, laplace, sem, kstest, spearmanr
from numpy.testing import assert_almost_equal


np.random.seed(9)


def test_stationary_validation():

    stationary = np.random.normal(size=1000)

    non_stationary = list()
    non_stationary.append(-1 if np.random.rand() < 0.5 else 1)
    for i in range(1, 1000):
        movement = -1 if np.random.rand() < 0.5 else 1
        value = non_stationary[i - 1] + movement
        non_stationary.append(value)

    has_stationary = np.array([stationary, non_stationary]).T
    has_no_stationary = np.array([non_stationary, non_stationary]).T

    results_stationary = bs._stationary(has_stationary)
    results_no_stationary = bs._stationary(has_no_stationary)

    assert results_stationary == [0] and results_no_stationary == []


def test_all_positive_validation():
    neg_series = np.random.randint(-100, 100)
    pos_series = np.random.randint(0, 100)

    has_negative = np.array([neg_series, pos_series]).T
    all_positive = np.array([pos_series, pos_series]).T

    assert bs._all_pos_values(all_positive) and not bs._all_pos_values(has_negative)


def test_removing_correlated_exogenous_columns():
    endog = np.random.randint(100, 200, size=(100, 5))
    ind_exog = np.random.randint(100, 200, size=(100, 2))
    exog = np.hstack((endog, ind_exog))

    sim = bs(endog, exogenous=exog, suppress_warnings=True)
    sim_exog = sim.exogenous

    assert np.all(ind_exog == sim_exog)


def test_merging_of_endogenous_and_exogenous_series():
    endog = np.zeros((100, 5))
    exog = np.ones((100, 2))

    merged = bs._merge_endog_and_exog(endog, exog)
    left_side = merged[:, :5]
    right_side = merged[:, -2:]

    assert (
        merged.shape == (100, 7)
        and np.all(left_side == endog)
        and np.all(right_side == exog)
    )


def test_removing_exogenous_from_merged_series():
    endog = np.zeros((100, 5))
    exog = np.ones((100, 2))

    merged = bs._merge_endog_and_exog(endog, exog)

    exog_removed = bs._remove_exog_from_merged(merged, exog)

    assert np.all(exog_removed == endog)


def test_log_pct_change():
    mv_series = np.array([[100, 130], [110, 140], [120, 150]])

    calculated = bs._log_change(mv_series)

    expected = np.array([[0.09531018, 0.07410797], [0.08701138, 0.06899287]])

    assert_almost_equal(calculated, expected)


def test_random_mean_generator():
    m = 0.0005
    s = 0.01
    n = 5000
    test_obs = laplace.rvs(loc=m, scale=s, size=n)

    se = sem(test_obs)
    test_dist = t(df=n - 1, loc=np.mean(test_obs), scale=se)
    lower_bound, upper_bound = test_dist.interval(alpha=0.7)
    random_test_means = list()

    while len(random_test_means) < n:
        random_mean = test_dist.rvs()
        if lower_bound <= random_mean <= upper_bound:
            random_test_means.append(random_mean)

    random_means_from_app = [bs._mean_rvs(test_obs) for _ in range(n)]

    _, kstest_pvalue = kstest(random_means_from_app, random_test_means)

    assert kstest_pvalue > 0.05  # null hypothesis: distributions are identical


def test_random_std_generator():
    m = 0.0005
    s = 0.01
    n = 5000
    test_obs = laplace.rvs(loc=m, scale=s, size=n)

    num = (n - 1) * (np.std(test_obs, ddof=1) ** 2)
    random_test_stds = np.sqrt(num / chi2.rvs(df=n - 1, size=n))

    random_stds_from_app = [bs._std_rvs(test_obs) for _ in range(n)]

    _, kstest_pvalue = kstest(random_stds_from_app, random_test_stds)

    assert kstest_pvalue > 0.05  # null hypothesis: distributions are identical


class TestSimulation:
    @pytest.fixture(scope="class")
    def gen_sims(self):
        obs = load_example_asset_price_history()
        bs_obj = bs(obs.values, suppress_warnings=True)
        sims = bs_obj.generate_simulations(
            n_sims=10,
            periods_per_sim=10_000,
            begin_values="norm",
            hmm_search_n_iter=60,
            hmm_search_n_fits_per_iter=5,
            hmm_search_params=None,
            output_as_float32=False,
            verbose=False,
        )
        return sims

    def test_number_of_simulations(self, gen_sims):
        sims = gen_sims
        assert sims.shape[0] == 10

    def test_periods_per_simulation(self, gen_sims):
        sims = gen_sims
        assert sims.shape[1] == 10_000

    def test_number_of_columns_maintained(self, gen_sims):
        obs = load_example_asset_price_history()
        sims = gen_sims
        assert sims.shape[2] == obs.shape[1]

    def test_simulation_correlation_similar_to_observed(self, gen_sims):
        obs = load_example_asset_price_history()
        obs_log_return = obs.pct_change().dropna().apply(np.log1p)
        obs_corr = obs_log_return.corr().values

        sims = gen_sims

        def coef(corr_matrix):
            # return upper triangle of correlation matrix
            return corr_matrix[np.triu_indices(corr_matrix.shape[0], k=1)]

        p_values = list()

        for sim in sims:
            sim = pd.DataFrame(sim)
            sim_log_return = sim.pct_change().dropna().apply(np.log1p)
            sim_corr = sim_log_return.corr().values

            _, p_value = spearmanr(coef(obs_corr), coef(sim_corr))
            p_values.append(p_value)

        p_values = np.array(p_values)

        assert np.all(p_values < 0.05)  # null hypothesis: two sets are uncorrelated

    def test_simulation_distributions_similar_to_observed(self, gen_sims):
        obs = load_example_asset_price_history()
        obs_log_return = obs.pct_change().dropna().apply(np.log1p).values

        n_cols = obs.shape[1]

        sims = gen_sims

        p_values = list()

        for sim in sims:
            sim = pd.DataFrame(sim)
            sim_log_return = sim.pct_change().dropna().apply(np.log1p).values

            for col in range(n_cols):
                _, p_value = kstest(obs_log_return[:, col], sim_log_return[:, col])
                p_values.append(p_value)

        p_values = np.array(p_values)

        assert np.all(p_values > 0.05)  # null hypothesis: distributions are identical
