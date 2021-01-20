import pprint
from collections import defaultdict, OrderedDict
from datetime import datetime

import multiprocessing as mp
import numpy as np
from joblib import delayed, Parallel
from functools import partial
from scipy.stats import poisson
from tqdm import tqdm
import copy
from data.processing.processing import get_data
from uncertainty.mcmc_utils import set_optimizer, compute_W, compute_B, accumulate, divide, divide_dict, avg_sum_chain, \
    avg_sum_multiple_chains, get_state, get_formatted_trials
import pdb
import scipy
from scipy.stats import norm as N
from scipy.stats import invgamma as inv
class MCMC(object):

    """
    MCMC class to perform metropolis-hastings. Supports gaussian and poisson likelihoods.
    
    Attributes:
        chains (list): List of chains where each chain contains a list of accepted and rejected parameters.
        df_district (pd.DataFrame): dataframe containing district data.
        df_train (pd.DataFrame): dataframe containing district data to train on.
        df_val (pd.DataFrame): dataframe containing district data to evaluate performance on.
        dist_log_likelihood (func): pointer to the log-likelihood function to use.
        fit_days (int): no. of days to evaluate performance on.
        fit_end (int): no. of days to fit the model on.
        fit_start (int): dataframe index to fit start day.
        R_hat (dict): dictionary containing the convergence metrics for each parameter.
        state (str): state that the district belongs to.
        timestamp (datetime.datetime): date and time when the model is run.
    """
    
    def __init__(self, optimiser, df_train, default_params, variable_param_ranges, n_chains, total_days,
 algo, num_evals, stride, proposal_sigmas, loss_method, loss_compartments, loss_indices,loss_weights, **ignored):
        """
        Constructor. Fetches the data, initializes the optimizer and sets up the
        likelihood function.
        
        Args:
            cfg (dict): dictonary containing all the config parameters.
            df_district (pd.DataFrame, optional): dataframe containing district data.
            **ignored: flag to ignore extra parameters.
        """
        self.timestamp = datetime.now()
        self.district = 'Mumbai'
        self.state = 'Maharashtra'
        self.stride = stride
        self.loss_weights = loss_weights
        #self.state = cfg['fitting']['data']['dataloading_params']['state']
        #self.district = cfg['fitting']['data']['dataloading_params']['district']
        #self.cfg  = cfg
        #self.df_district = df_district if df_district is not None else self._fetch_data()
        self.loss_indices = loss_indices
        self.df_train  = df_train[loss_indices[0]:loss_indices[1]]
        self.loss_method = loss_method
        #self.train_days = self.cfg['fitting']['split']['train_period']
        self.n_chains =  n_chains
        self.total_days = total_days
        self.likelihood = algo
        self._default_params = default_params
        self.prior_ranges = variable_param_ranges
        self.iters = num_evals
        self.compartments = loss_compartments
        self._optimiser = optimiser
        #self._optimiser, self._default_params = set_optimizer(self.df_train, df_train.size(), 
        #                                    self._default_params_old, optimiser)
        self.proposal_sigmas = proposal_sigmas
        self.dist_log_likelihood = eval("self._{}_log_likelihood".format(self.likelihood))
        self.DIC = 0

    def _fetch_data(self):
        """
        Fetches district data from either tracker or athena database.
        
        Returns:
            pd.DataFrame: dataframe containing district data.
        """
        cfg = self.cfg
        data_source = cfg['fitting']['data']['data_source']
        dataloading_params = cfg['fitting']['data']['dataloading_params']

        return get_data(data_source,dataloading_params)

    def _split_data(self):
        """
        Splits self.df_district into train and val set as per number of test days specified
        in the config.
        """
        self.test_days = self.cfg['fitting']['split']['val_period']
        self.train_days = self.cfg['fitting']['split']['train_period']
        self.end_date = self.cfg['fitting']['split']['end_date']
        N = len(self.df_district)
        drop = (self.df_district[-1:]['date'].item().date() - self.end_date).days
        self.df_district = self.df_district[:N-drop]
        N = len(self.df_district)
        self.df_val = self.df_district.iloc[N - self.test_days:, :]
        self.fit_days = N - self.test_days

        self.fit_start = N - self.test_days - self.train_days
        self.fit_end = N - self.test_days

        self.df_train = self.df_district.iloc[self.fit_start : self.fit_end, :]
        print("Train set:\n", self.df_train)
        print("Val set:\n", self.df_val)

    def _param_init(self):
        """
        Samples the first value for each param from a uniform prior distribution with ranges
        mentioned in the config.
        
        Returns:
            dict: dictionary containing initial param-value pairs.
        """
        theta = defaultdict()
        for key in self.prior_ranges:
            theta[key] = np.random.uniform(float(self.prior_ranges[key][0][0]), self.prior_ranges[key][0][1])
        oldLL,da,db = self._log_likelihood(theta)
        alpha = 40
        beta =  2/700
        from scipy.stats import invgamma as inv
        theta['gamma'] = np.sqrt(inv.rvs(a = alpha ,scale = beta , size = 1 )[0])
        return theta,oldLL,da,db

    def Gauss_proposal(self, theta_old):
        """
        Proposes a new set of parameter values given the previous set.
        
        Args:
            theta_old (dict): dictionary of old param-value pairs.
        
        Returns:
            OrderedDict: dictionary of newly proposed param-value pairs.
        """
        theta_new = OrderedDict()
        
        for param in theta_old:
            if param == 'gamma':
                theta_new[param] = theta_old[param]
                continue
            old_value = theta_old[param]
            new_value = None
            lower_bound = (float(self.prior_ranges[param][0][0]))
            upper_bound = (self.prior_ranges[param][0][1])
            #print(lower_bound, upper_bound)
            while new_value == None:
                #Gaussian proposal
                new_value = np.random.normal(loc = old_value, scale = self.proposal_sigmas[param])
                # new_value = np.random.normal(loc=np.exp(old_value), scale=np.exp((self.proposal_sigmas[param])))
                if new_value < lower_bound or new_value > upper_bound:
                    new_value = None
                    continue
                # new_value = np.log(new_value)
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
        diff = np.sum(((true - pred) ** 2))
        ll = - (N * np.log(np.sqrt(2*np.pi) * sigma)) - (diff / (2 * sigma ** 2))
        return ll, N/2 ,diff/2

    def _poisson_log_likelihood(self, true, pred, *ignored):
        """
        Calculates the log-likelihood of the data as per a poisson distribution.
        
        Args:
            true (np.ndarray): array containing actual numbers.
            pred (np.ndarray): array containing estimated numbers.
        
        Returns:
            float: poisson log-likelihood of the data.
        """
        T = np.ediff1d(true)
        P = np.ediff1d(pred)
        ll = np.log(poisson.pmf(k = T, mu = P))
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
        da = 0
        db = 0
        params_dict = {**theta, **self._default_params}
        df_prediction = self._optimiser.solve(params_dict,end_date = self.df_train[-1:]['date'].item())
        sigma = theta ['gamma']

        for compartment in self.compartments:
            pred = np.array(df_prediction[compartment], dtype=np.int64)
            true = np.array(self.df_train[compartment], dtype=np.int64)
            T = np.ediff1d(np.log(true))
            P = np.ediff1d(np.log(pred))
            LL,d_alpha,d_beta = self.dist_log_likelihood(T, P, sigma)
            ll += LL
            da += d_alpha
            db += d_beta
        return ll,da,db

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

    def truncated_gaussian(self,theta):
        multiplier_offset = 1
        for param in theta:
            if param == 'gamma':
                continue
            a = (float(self.prior_ranges[param][0][0]))
            b = (self.prior_ranges[param][0][1])
            sigma = self.proposal_sigmas[param]
            param_value = theta[param]
            multiplier_offset *= ( N.cdf((b-param_value)/sigma) -  N.cdf((a-param_value)/sigma) )
        return multiplier_offset

    def _accept(self, theta_old, theta_new, explored , optimized,oldLL):    
        """
        Decides, using the likelihood function, whether a newly proposed set of param values should 
        be accepted and added to the MCMC chain.
        
        Args:
            theta_old (OrderedDict): previous parameter-value pairs.
            theta_new (OrderedDict): new proposed parameter-value pairs.
        
        Returns:
            bool: whether or not to accept the new parameter set.
        """
        x_new,da,db= self._log_likelihood(theta_new)
        x_old,_,_ = self._log_likelihood(theta_old)

        if (x_new) > (x_old):
            optimized+=1
            return True,explored,optimized,x_new,da,db
        else:
            x = np.random.uniform(0, 1)
            alpha = (np.exp(x_new) * self.truncated_gaussian(theta_old)) / (np.exp(x_old) * self.truncated_gaussian(theta_new) )
            accept_bool = x < alpha
            if accept_bool :
                explored+=1
            return accept_bool,explored,optimized,x_new,da,db

    def _metropolis(self, iters):
        """
        Implementation of the metropolis loop.
        
        Returns:
            tuple: tuple containing a list of accepted and a list of rejected parameter values.
        """
        theta,oldLL,da,db = self._param_init()
        accepted = [theta]
        rejected = list()
        A = 0
        accepted_LIK = [oldLL]
        explored = 0
        optimized = 0
        alpha_0 = 40
        beta_0 = 2/700
        scipy.random.seed()
        for i in tqdm(range(iters)):
            
            theta_new = self.Gauss_proposal(theta)
            theta_new['gamma'] = np.sqrt(inv.rvs(a = alpha_0 + da  ,scale = beta_0 + db,size = 1)[0])
            
            accept_choice,explored , optimized, LL , da, db = self._accept(theta, theta_new, explored, optimized,oldLL)
            if accept_choice:
                oldLL = LL
                theta = copy.deepcopy(theta_new)
                A += 1
            else:
                theta['gamma'] = theta_new['gamma']
                rejected.append(theta_new)
            accepted_LIK.append(oldLL)
            accepted.append(copy.deepcopy(theta))
        print("The acceptance ratio is --------> ",A / iters )
        print("The explored steps are --------> ",explored )
        print("The optimized steps are --------> ",optimized )
        return accepted, rejected , accepted_LIK

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
        combined_LIK = list()
        for k, run in enumerate(self.chains):
            burn_in = int(len(run) / 2)
            combined_acc += run[0][burn_in:][::self.stride]
            combined_LIK += run[2][burn_in:][::self.stride]
        n_samples = 2000
        sample_indices = np.random.uniform(0, len(combined_acc), n_samples)
        #Total day is 1 less than training_period
        sample_indices = [int(i) for i in sample_indices]
        selected_LIK = [combined_LIK[i] for i in sample_indices]
        selected_ACC = [combined_acc[i] for i in sample_indices]
        losses = list()
        params = list()
        for i in tqdm(sample_indices):
            params.append(combined_acc[int(i)])
            losses.append(self._optimiser.solve_and_compute_loss(combined_acc[int(i)],self._default_params,
                                                                 self.df_train, self.total_days,
                                                                 loss_indices=self.loss_indices,
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
        paralell_bool = True
        
        if paralell_bool:
            print("Executing in Parallel")
            partial_metropolis = partial(self._metropolis, iters=self.iters)
            self.chains = Parallel(n_jobs=self.n_chains)(
                delayed(partial_metropolis)() for _ in range(self.n_chains))
        else:
            print("Executing in Serial")
            self.chains = []
            for i, run in enumerate(range(self.n_chains)):
                self.chains.append(self._metropolis(self.iters))
        self._check_convergence()
        return self._get_trials()
