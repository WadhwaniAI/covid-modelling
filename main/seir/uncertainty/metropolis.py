from datetime import datetime
from collections import defaultdict, OrderedDict

import pandas as pd
import numpy as np
from joblib import delayed, Parallel

from main.seir.uncertainty import Uncertainty
from utils.fitting.loss import Loss_Calculator


class MCMCUncertainty(Uncertainty):

    def __init__(self, region_dict, date_of_interest):
        super().__init__(region_dict)
        self.trials = None
        self.date_of_interest = datetime.strptime(date_of_interest, '%Y-%m-%d')
        self.get_distribution()

    def get_distribution(self):
        df = pd.DataFrame(columns=[self.date_of_interest, 'cdf'])
        df[self.date_of_interest] = self.region_dict['m2']['all_trials'].loc[:, self.date_of_interest]

        df = df.sort_values(by=self.date_of_interest)
        df.index.name = 'idx'
        df.reset_index(inplace=True)

        df['cdf'] = [(x*100)/len(df['idx']) for x in range(1, len(df['idx'])+1)]

        self.distribution = df
        return self.distribution

    def _param_init(self):
        """
        Samples the first value for each param from a uniform prior distribution with ranges
        mentioned in the config.
        
        Returns:
            dict: dictionary containing initial param-value pairs.
        """
        theta = defaultdict()
        for key in self.prior_ranges:
            theta[key] = np.random.uniform(self.prior_ranges[key][0], self.prior_ranges[key][1])
            
        return theta

    def _proposal(self, theta_old):
        """
        Proposes a new set of parameter values given the previous set.
        
        Args:
            theta_old (dict): dictionary of old param-value pairs.
        
        Returns:
            OrderedDict: dictionary of newly proposed param-value pairs.
        """
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
        """
        Calculates the log-likelihood of the data as per a gaussian distribution.
        
        Args:
            true (np.ndarray): array containing actual numbers.
            pred (np.ndarray): array containing estimated numbers.
            sigma (float): standard deviation of the gaussian distribution to use.
        
        Returns:
            float: gaussian log-likelihood of the data.
        """
        N = len(true)
        ll = - (N * np.log(np.sqrt(2*np.pi) * sigma)) - (np.sum(((true - pred) ** 2) / (2 * sigma ** 2)))
        return ll

    def _poisson_log_likelihood(self, true, pred, **ignored):
        """
        Calculates the log-likelihood of the data as per a poisson distribution.
        
        Args:
            true (np.ndarray): array containing actual numbers.
            pred (np.ndarray): array containing estimated numbers.
        
        Returns:
            float: poisson log-likelihood of the data.
        """
        ll = np.log(poisson.pmf(k = pred, mu = true.mean()))
        return np.sum(ll)

    def _log_likelihood(self, theta):
        """
        Computes and returns the data log-likelihood as per either a gaussian or poisson distribution.
        
        Args:
            theta (OrderedDict): parameter-value pairs.
        
        Returns:
            float: log-likelihood of the data.
        """
        ll = 0
        df_prediction = self._optimiser.solve(theta, self._default_params, self.df_train)
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
        """
        Computes and returns the log-prior as per non-negative constraint on the parameters.
        
        Args:
            theta (OrderedDict): parameter-value pairs.
        
        Returns:
            float: log-prior of the parameters.
        """
        if (np.array([*theta.values()]) < 0).any():
            prior = 0
        else:
            prior = 1
        
        return np.log(prior)

    def _accept(self, theta_old, theta_new):    
        """
        Decides, using the likelihood function, whether a newly proposed set of param values should 
        be accepted and added to the MCMC chain.
        
        Args:
            theta_old (OrderedDict): previous parameter-value pairs.
            theta_new (OrderedDict): new proposed parameter-value pairs.
        
        Returns:
            bool: whether or not to accept the new parameter set.
        """
        x_new = self._log_likelihood(theta_new)
        x_old = self._log_likelihood(theta_old)
        
        if (x_new) > (x_old):
            return True
        else:
            x = np.random.uniform(0, 1)
            return (x < np.exp(x_new - x_old))

    def _metropolis(self):
        """
        Implementation of the metropolis loop.
        
        Returns:
            tuple: tuple containing a list of accepted and a list of rejected parameter values.
        """
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
        """
        Computes the Gelman-Rubin convergence statistics for each parameter and stores them in
        self.R_hat.
        """
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

    def _get_trials(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        combined_acc = list()
        for k, run in enumerate(self.chains):
            burn_in = int(len(run) / 2)
            combined_acc += run[0][burn_in:]

        n_samples = 1000
        sample_indices = np.random.uniform(0, len(combined_acc), n_samples)

        total_days = len(self.df_train['date'])
        loss_indices = [-self.fit_days, None]

        losses = list()
        params = list()
        for i in tqdm(sample_indices):
            params.append(combined_acc[int(i)])
            losses.append(self._optimiser.solve_and_compute_loss(combined_acc[int(i)], self._default_params,
                                                                 self.df_train, total_days,
                                                                 loss_indices=loss_indices,
                                                                 loss_method=self.loss_method))

        least_loss_index = np.argmin(losses)
        best_params = params[int(least_loss_index)]
        trials = get_formatted_trials(params, losses)
        return best_params, trials

    def run(self):
        """
        Runs all the metropolis-hastings chains in parallel.
        
        Returns:
            list: Description
        """
        self.chains = Parallel(n_jobs=self.n_chains)(delayed(self._metropolis)() for i, run in enumerate(range(self.n_chains)))
        self._check_convergence()
        return self._get_trials()
