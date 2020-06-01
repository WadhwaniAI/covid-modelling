import pprint
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from datetime import datetime
from joblib import delayed, Parallel
from collections import defaultdict, OrderedDict

from data.processing import get_district_time_series
from data.dataloader import get_covid19india_api_data
from mcmc_utils import set_optimizer, compute_W, compute_B, accumulate, divide, divide_dict, avg_sum_chain, avg_sum_multiple_chains


class MCMC(object):
    def __init__(self, state: str, district: str, n_chains: int, iters: int, fit_days: int = 10, test_days: int = 5, fit2new: bool = False):
        self.timestamp = datetime.now()
        self.state = state
        self.district = district
        self.n_chains = n_chains
        self.fit_days = fit_days
        self.test_days = test_days
        self._iters = iters
        self._fit2new = fit2new

        self._fetch_data()
        self._split_data()
        self._set_prior_ranges()
        self._set_proposal_sigmas()
        self._optimiser, self._default_params = set_optimizer(self.df_train)

    def _fetch_data(self):
        dataframes = get_covid19india_api_data()
        self.df_district = get_district_time_series(dataframes, state=self.state, district=self.district, use_dataframe = 'districts_daily')

    def _split_data(self):
        self.df_train = self.df_district.iloc[:-self.test_days, :]
        self.df_val = self.df_district.iloc[-self.test_days:, :]

    def _get_new_cases_array(self, cases):
        return np.array(cases[1:]) - np.array(cases[:-1])

    def _set_prior_ranges(self):
        self.prior_ranges = OrderedDict()
        self.prior_ranges['R0'] = (1.6, 5)
        self.prior_ranges['T_inc'] = (4, 5)
        self.prior_ranges['T_inf'] = (3, 4)
        self.prior_ranges['T_recov_severe'] = (9, 20)
        self.prior_ranges['P_severe'] = (0.3, 0.99)
        self.prior_ranges['P_fatal'] = (1e-10, 0.3)
        self.prior_ranges['intervention_amount'] = (1e-10, 1)
        self.prior_ranges['sigma'] = (0, 1)

    def _param_init(self):
        theta = defaultdict()
        for key in self.prior_ranges:
            theta[key] = np.random.uniform(self.prior_ranges[key][0], self.prior_ranges[key][1])
            
        return theta

    def _set_proposal_sigmas(self):
        self.proposal_sigmas = OrderedDict()
        self.proposal_sigmas['R0'] = 20
        self.proposal_sigmas['T_inc'] = np.exp(10)
        self.proposal_sigmas['T_inf'] = np.exp(10)
        self.proposal_sigmas['T_recov_severe'] = np.exp(100)
        self.proposal_sigmas['P_severe'] = 10
        self.proposal_sigmas['P_fatal'] = 10
        self.proposal_sigmas['intervention_amount'] = 10
        self.proposal_sigmas['sigma'] = 10

    def _proposal(self, theta_old):
        theta_new = [np.nan]
        
        while np.isnan(theta_new).any():
            theta_new = np.random.normal(loc=np.exp([*theta_old.values()]), scale=[*self.proposal_sigmas.values()])
            theta_new = np.log(theta_new)
        
        return dict(zip(theta_old.keys(), theta_new))

    def _log_likelihood(self, theta):
        if (np.array([*theta.values()]) < 0).any():
            return -np.inf
        df_prediction = self._optimiser.solve(theta, self._default_params, self.df_train)
        pred = np.array(df_prediction['total_infected'].iloc[-self.fit_days:])
        true = np.array(self.df_train['total_infected'].iloc[-self.fit_days:])
        if self._fit2new:
            pred = self._get_new_cases_array(pred.copy())
            true = self._get_new_cases_array(true.copy())
        sigma = theta['sigma']
        N = len(true)
        ll = - (N * np.log(np.sqrt(2*np.pi) * sigma)) - (np.sum(((true - pred) ** 2) / (2 * sigma ** 2)))
        return ll

    def _log_prior(self, theta):
        if (np.array([*theta.values()]) < 0).any():
            prior = 0
        else:
            prior = 1
        
        return np.log(prior)

    def _accept(self, theta_old, theta_new):    
        x_new = self._log_likelihood(theta_new) + self._log_prior(theta_new)
        x_old = self._log_likelihood(theta_old) + self._log_prior(theta_old)
        
        if (x_new) > (x_old):
            return True
        else:
            x = np.random.uniform(0, 1)
            return (x < np.exp(x_new - x_old))

    def _metropolis(self):
        theta = self._param_init()
        accepted = [theta]
        rejected = list()
        
        for i in tqdm(range(self._iters)):
            theta_new = self._proposal(theta)
            if self._accept(theta, theta_new):
                theta = theta_new
            else:
                rejected.append(theta_new)
            accepted.append(theta)
        
        return accepted, rejected

    def _check_convergence(self):
        accepted_chains = [chain[0] for chain in self.chains]
        burn_in = int(len(accepted_chains[0]) / 2)
        sampled_chains = [chain[:burn_in] for chain in accepted_chains]
        split_chains = [sampled_chain[int(burn_in/2):] for sampled_chain in sampled_chains] \
                       + [sampled_chain[:int(burn_in/2)] for sampled_chain in sampled_chains]

        chain_sums_avg = []
        for chain in split_chains:
            chain_sums_avg.append(avg_sum_chain(chain))
        multiple_chain_sums_avg = avg_sum_multiple_chains(chain_sums_avg)

        m = len(split_chains)
        n = len(split_chains[0])
        W =  compute_W(split_chains, chain_sums_avg, n, m)
        B =  compute_B(multiple_chain_sums_avg, chain_sums_avg, n, m)
        var_hat = accumulate([divide(W, n/(n-1)), divide(B, n) ])
        R_hat_sq = divide_dict(var_hat, W)
        R_hat = {key:np.sqrt(value) for key, value in R_hat_sq.items()}
        neff = divide_dict(var_hat, B)
        neff = {key: m*n*value for key, value in neff.items()}

        print("Gelman-Rubin convergence statistics (variance ratios):")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(R_hat)

    def run(self):
        self.chains = Parallel(n_jobs=self.n_chains)(delayed(self._metropolis)() for run in range(self.n_chains))
        self._check_convergence()

