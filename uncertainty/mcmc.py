import pprint
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from datetime import datetime
from scipy.stats import poisson
from joblib import delayed, Parallel
from collections import defaultdict, OrderedDict

from data.processing import get_district_time_series
from mcmc_utils import set_optimizer, compute_W, compute_B, accumulate, divide, divide_dict, avg_sum_chain, avg_sum_multiple_chains, get_state


class MCMC(object):
    def __init__(self, cfg, **ignored):
        self.timestamp = datetime.now()
        self.state = get_state(cfg["district"])
        self.__dict__.update(cfg)

        self._fetch_data()
        self._split_data()
        self._optimiser, self._default_params = set_optimizer(self.df_train)
        self.dist_log_likelihood = eval("self._{}_log_likelihood".format(self.likelihood))

    def _fetch_data(self):
        self.df_district = get_district_time_series(state=self.state, district=self.district, use_dataframe = 'raw_data')

    def _split_data(self):
        N = len(self.df_district)
        self.df_val = self.df_district.iloc[N - self.test_days:, :]
        if self.fit_days is None:
            self.fit_days = N - self.test_days

        self.fit_start = N - self.test_days - self.fit_days
        self.fit_end = N - self.test_days

        self.df_train = self.df_district.iloc[self.fit_start : self.fit_end, :]
        print("Train set:\n", self.df_train)
        print("Val set:\n", self.df_val)

    def _get_new_cases_array(self, cases):
        return np.array(cases[1:]) - np.array(cases[:-1])

    def _param_init(self):
        theta = defaultdict()
        for key in self.prior_ranges:
            theta[key] = np.random.uniform(self.prior_ranges[key][0], self.prior_ranges[key][1])
            
        return theta

    def _proposal(self, theta_old):
        theta_new = OrderedDict()
        
        for param in theta_old:
            old_value = theta_old[param]
            new_value = -1
            while np.isnan(new_value) or new_value <=0:
                new_value = np.random.normal(loc=np.exp(old_value), scale=np.exp(self.proposal_sigmas[param]))
                new_value = np.log(new_value)
            theta_new[param] = new_value
        
        return theta_new

    def _gaussian_log_likelihood(self, true, pred, sigma):
        N = len(true)
        ll = - (N * np.log(np.sqrt(2*np.pi) * sigma)) - (np.sum(((true - pred) ** 2) / (2 * sigma ** 2)))
        return ll

    def _poisson_log_likelihood(self, true, pred, sigma):
        ll = np.log(poisson.pmf(k = pred, mu = true.mean()))
        return np.sum(ll)

    def _log_likelihood(self, theta):
        ll = 0
        df_prediction = self._optimiser.solve(theta, self._default_params, self.df_train, initialisation='intermediate', loss_indices=[0, -1])
        sigma = theta['sigma']

        for compartment in self.compartments:
            pred = np.array(df_prediction[compartment], dtype=np.int64)
            true = np.array(self.df_train[compartment], dtype=np.int64)
            if self.fit2new:
                pred = self._get_new_cases_array(pred.copy())
                true = self._get_new_cases_array(true.copy())

            ll += self.dist_log_likelihood(true, pred, sigma)

        return ll

    def _log_prior(self, theta):
        if (np.array([*theta.values()]) < 0).any():
            prior = 0
        else:
            prior = 1
        
        return np.log(prior)

    def _accept(self, theta_old, theta_new):    
        x_new = self._log_likelihood(theta_new)
        x_old = self._log_likelihood(theta_old)
        
        if (x_new) > (x_old):
            return True
        else:
            x = np.random.uniform(0, 1)
            return (x < np.exp(x_new - x_old))

    def _metropolis(self, seed):
        # np.random.seed(seed)
        theta = self._param_init()
        accepted = [theta]
        rejected = list()
        
        for i in tqdm(range(self.iters)):
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
        self.R_hat = R_hat

    def run(self):
        self.chains = Parallel(n_jobs=self.n_chains)(delayed(self._metropolis)(100*i) for i, run in enumerate(range(self.n_chains)))
        self._check_convergence()

