import copy
import pprint
from collections import OrderedDict, defaultdict
from datetime import datetime
from functools import partial

import numpy as np
from joblib import Parallel, delayed
from main.seir.uncertainty.base import Uncertainty
from scipy.stats import invgamma as inv
from scipy.stats import norm as N
from tqdm import tqdm

class MCMC(Uncertainty):

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
    
    def __init__(self, df_train, default_params, variable_param_ranges, n_chains, total_days,
                 algo, num_evals,num_samples, stride, proposal_sigmas, loss_method, loss_compartments, loss_indices,
                 loss_weights, model,partial_predict,partial_predict_and_compute_loss, **ignored):
        """
        Constructor. Fetches the data, initializes the optimizer and sets up the
        likelihood function.
        
        Args:
            cfg (dict): dictonary containing all the config parameters.
            df_district (pd.DataFrame, optional): dataframe containing district data.
            **ignored: flag to ignore extra parameters.
        """
        self.timestamp = datetime.now()
        self.stride = stride
        self.loss_weights = loss_weights
        self.loss_indices = loss_indices
        self.df_train  = df_train[loss_indices[0]:loss_indices[1]]
        self.loss_method = loss_method
        self.n_chains =  n_chains
        self.total_days = total_days
        self.partial_predict = partial_predict
        self.partial_predict_and_compute_loss = partial_predict_and_compute_loss
        self.likelihood = algo
        self._default_params = default_params
        self.prior_ranges = variable_param_ranges
        self.num_samples = num_samples
        self.num_evals = num_evals
        self.compartments = loss_compartments
        self.model = model
        self.proposal_sigmas = proposal_sigmas
        self.dist_log_likelihood = eval("self._{}_log_likelihood".format(self.likelihood))
        self.DIC = 0
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
        _,da,db = self._log_likelihood(theta)
        alpha = 40
        beta =  2/700
        theta['gamma'] = np.sqrt(inv.rvs(a = alpha ,scale = beta , size = 1 )[0])
        return theta,da,db

    def Gauss_proposal(self, theta_old):
        """
        Proposes a new set of parameter values given the previous set using gaussian noise.
        
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
            while new_value is None:
                #Gaussian proposal
                new_value = np.random.normal(loc = old_value, scale = self.proposal_sigmas[param])
                if new_value < lower_bound or new_value > upper_bound:
                    new_value = None
                    continue
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
        df_prediction = self.partial_predict(params_dict)
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

    def truncated_gaussian(self,theta):
        """
        Calculates Multiplier offset to take care of truncated gaussian
        
        Args:
            theta(OrderedDict): set of parameters
        
        Returns:
            float: Multiplier offset to take care of truncated gaussian 
        """
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

    def _accept(self, theta_old, theta_new, explored , optimized):    
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

        if x_new > x_old:
            optimized+=1
            return True,explored,optimized,x_new,da,db
        else:
            x = np.random.uniform(0, 1)
            alpha = (np.exp(x_new) * self.truncated_gaussian(theta_old)) / (np.exp(x_old) * self.truncated_gaussian(theta_new) )
            accept_bool = x < alpha
            if accept_bool :
                explored+=1
            return accept_bool,explored,optimized,x_new,da,db

    def _metropolis(self, iters, A=0, alpha_0=40, beta_0=2/700, seed=None):
        """
        Implementation of the metropolis loop.
        
        Returns:
            tuple: tuple containing a list of accepted and a list of rejected parameter values.
        """
        theta,da,db = self._param_init()
        accepted = [theta]
        rejected = list()
        explored = 0
        optimized = 0
        np.random.seed(seed=seed)
        for _ in tqdm(range(iters)):
            
            theta_new = self.Gauss_proposal(theta)
            theta_new['gamma'] = np.sqrt(inv.rvs(a = alpha_0 + da  ,scale = beta_0 + db,size = 1)[0])
            
            accept_choice,explored , optimized, _ , da, db = self._accept(theta, theta_new, explored, optimized)
            if accept_choice:
                theta = copy.deepcopy(theta_new)
                A += 1
            else:
                theta['gamma'] = theta_new['gamma']
                rejected.append(theta_new)
            accepted.append(copy.deepcopy(theta))
        print("The acceptance ratio is --------> ",A / iters )
        print("The explored steps are --------> ",explored )
        print("The optimized steps are --------> ",optimized )
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
            chain_sums_avg.append(self.avg_sum_chain(chain))
        multiple_chain_sums_avg = self.avg_sum_multiple_chains(chain_sums_avg)

        m = len(split_chains)
        n = len(split_chains[0])
        W =  self.compute_W(split_chains, chain_sums_avg, n, m)
        B =  self.compute_B(multiple_chain_sums_avg, chain_sums_avg, n, m)
        var_hat = self.accumulate([self.divide(W, n/(n-1)), self.divide(B, n) ])
        R_hat_sq = self.divide_dict(var_hat, W)
        R_hat = {key:np.sqrt(value) for key, value in R_hat_sq.items()}
        neff = self.divide_dict(var_hat, B)
        neff = {key: m*n*value for key, value in neff.items()}

        print("Gelman-Rubin convergence statistics (variance ratios):")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(R_hat)
        self.R_hat = R_hat

    def get_formatted_trials(self,params, losses):
        """Summary
        
        Args:
            params (TYPE): Description
            losses (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        trials = list()
        for i, param_dict in enumerate(params):
            trial = {'result': dict(), 'misc': dict()}
            trial['result']['loss'] = losses[i]
            trial['misc']['vals'] = {k: [v] for k, v in param_dict.items()}
            trials.append(trial)
        return trials

    def _get_trials(self):
        """Summary      
        Modifies samples collected from MCMC to mimic trials object generated from BO
        
        Returns:
            TYPE: Description
        """
        combined_acc = list()
        for _, run in enumerate(self.chains):
            burn_in = int(len(run) / 2)
            combined_acc += run[0][burn_in:][::self.stride]
        n_samples = self.num_samples
        sample_indices = np.random.uniform(0, len(combined_acc), n_samples)
        #Total day is 1 less than training_period
        sample_indices = [int(i) for i in sample_indices]
        losses = list()
        params = list()
        for i in tqdm(sample_indices):
            params.append(combined_acc[int(i)])
            losses.append(self.partial_predict_and_compute_loss(combined_acc[int(i)]))

        least_loss_index = np.argmin(losses)
        best_params = params[int(least_loss_index)]
        trials = self.get_formatted_trials(params, losses)
        return best_params, trials

    def run(self, parallelise=False):
        """
        Runs all the metropolis-hastings chains in parallel/serial.
        
        Returns:
            list: best params and set of trials mimicing bayesopt trials object
        """
        
        if parallelise:
            print("Executing in Parallel")
            partial_metropolis = partial(self._metropolis, iters=self.iters)
            self.chains = Parallel(n_jobs=self.n_chains)(
                delayed(partial_metropolis)() for _ in range(self.n_chains))
        else:
            print("Executing in Serial")
            chains = []
            for _ in range(self.n_chains):
                chains.append(self._metropolis(self.num_samples))

            self.chains = chains
        self._check_convergence()
        return self._get_trials()

    def get_PI(self,pred_dfs, date, key, multiplier=1.96):
        """Summary
        
        Args:
            pred_dfs (TYPE): Description
            date (TYPE): Description
            key (TYPE): Description
            multiplier (float, optional): Description
        
        Returns:
            TYPE: Description
        """
        pred_samples = list()
        for df in pred_dfs:
            pred_samples.append(df.loc[date, key])
            
        mu = np.array(pred_samples).mean()
        sigma = np.array(pred_samples).std()
        low = mu - multiplier*sigma
        high = mu + multiplier*sigma
        return mu, low, high

    def accumulate(self,dict_list):
        """Summary
        
        Args:
            dict_list (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        accumulator = defaultdict(int)
        for elt in dict_list:
            for key in elt:
                accumulator[key]+=elt[key]
        return accumulator

    def divide(self,dictvar, num):
        """Summary
        
        Args:
            dictvar (TYPE): Description
            num (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        return {key:dictvar[key]/num for key in dictvar}

    def avg_sum_chain(self,chain):
        """Summary
        
        Args:
            chain (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        chain_sums_avg = self.accumulate(chain)
        return self.divide(chain_sums_avg, len(chain))

    def avg_sum_multiple_chains(self,chain_sums_avg):
        """Summary
        
        Args:
            chain_sums_avg (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        multiple_chain_sums_avg = self.accumulate(chain_sums_avg)
        return self.divide(multiple_chain_sums_avg, len(chain_sums_avg))

    def compute_B(self,multiple_chain_sums_avg, chain_sums_avg, n, m):
        """Summary
        
        Args:
            multiple_chain_sums_avg (TYPE): Description
            chain_sums_avg (TYPE): Description
            n (TYPE): Description
            m (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        B = defaultdict(int)
        for elt in chain_sums_avg:
            for key in elt:
                B[key] += np.square(elt[key] - multiple_chain_sums_avg[key])
        return self.divide(B, (m-1)/n)

    def compute_W(self,split_chains, chain_sums_avg, n, m):
        """Summary
        
        Args:
            split_chains (TYPE): Description
            chain_sums_avg (TYPE): Description
            n (TYPE): Description
            m (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        s = []
        for j in range(m):
            s_j_sq = defaultdict(int)
            chain = split_chains[j]
            chain_sum_avg_j = chain_sums_avg[j]
            for i in range(n):
                chain_elt = chain[i]
                for key in chain_elt:
                    s_j_sq[key] += np.square(chain_elt[key] - chain_sum_avg_j [key])
            s_j_sq = self.divide(s_j_sq, n - 1)
            s.append(s_j_sq)
        return (self.divide (self.accumulate(s),m))

    def divide_dict(self,d1, d2):
        """Summary
        
        Args:
            d1 (TYPE): Description
            d2 (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        accumulator = defaultdict(int)
        for key in d1:
            accumulator[key] = d1[key]/d2[key]
        return accumulator
