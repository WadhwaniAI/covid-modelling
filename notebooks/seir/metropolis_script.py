import os
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
import pickle
matplotlib.rcParams.update({'font.size': 18})
from joblib import delayed, Parallel
from collections import defaultdict, OrderedDict

from utils.generic import init_params
from main.seir.optimiser import Optimiser
from models.seir.seir_testing import SEIR_Testing
from data.processing import get_district_time_series
from data.dataloader import get_covid19india_api_data


def get_data(district, state):
    dataframes = get_covid19india_api_data()
    df_district = get_district_time_series(dataframes, state=state, district=district)

    df_train = df_district.iloc[:-5, :]
    df_val = df_district.iloc[-5:, :]

    return df_district, df_train, df_val

def get_prior_ranges():
    ## assuming uniform priors, following dictionary contains the ranges
    prior_ranges = OrderedDict()
    prior_ranges['R0'] = (1, 3)#(1.6, 3)
    prior_ranges['T_inc'] = (1, 5) #(4, 5)
    prior_ranges['T_inf'] = (1, 4) #(3, 4)
    prior_ranges['T_recov_severe'] = (9, 20)
    prior_ranges['P_severe'] = (0.3, 0.99)
    prior_ranges['intervention_amount'] = (0.3, 1)
    prior_ranges['sigma'] = (0, 1)

    return prior_ranges

def param_init(prior_ranges):
    theta = defaultdict()
    for key in prior_ranges:
        theta[key] = np.random.uniform(prior_ranges[key][0], prior_ranges[key][1])
        
    return theta

def get_proposal_sigmas(prior_ranges):
    proposal_sigmas = OrderedDict()
    for key in prior_ranges:
        proposal_sigmas[key] = 0.025 * (prior_ranges[key][1] - prior_ranges[key][0])

    return proposal_sigmas

def proposal(theta_old, proposal_sigmas):
    theta_new = np.random.normal(loc=[*theta_old.values()], scale=[*proposal_sigmas.values()])
    return dict(zip(theta_old.keys(), theta_new))

def log_likelihood(theta, df_train, fit_days=10):
    if any(value < 0 for value in theta.values()):
        return -np.inf
    theta_model = theta.copy()
    del theta_model['sigma']
    optimiser = Optimiser()
    default_params = optimiser.init_default_params(df_train)
    df_prediction = optimiser.solve(theta_model, default_params, df_train)
    pred = np.array(df_prediction['total_infected'])[-fit_days:]
    true = np.array(df_train['total_infected'])[-fit_days:]
    sigma = theta['sigma']
    N = len(true)
    try:
        ll = - (N * np.log(np.sqrt(2*np.pi) * sigma)) - (np.sum(((true - pred) ** 2) / (2 * sigma ** 2)))
    except ValueError:
        print("ValueError at params: ", theta_model)
        return -np.inf

    return ll

def log_prior(theta):
    if any(value < 0 for value in theta.values()):
        prior = 0
    else:
        prior = 1
    
    return np.log(prior)

def in_valid_range(key, value):
    return (value <= prior_ranges[key][1]) and (value >= prior_ranges[key][0])

def accept(theta_old, theta_new, df_train):    
    x_new = log_likelihood(theta_new, df_train) + log_prior(theta_new)
    x_old = log_likelihood(theta_old, df_train) + log_prior(theta_old)
    
    if (x_new) > (x_old):
        return True
    else:
        x = np.random.uniform(0, 1)
        return (x < np.exp(x_new - x_old))
    
def anneal_accept(iter):
    prob = 1 - np.exp(-(1/(iter + 1e-10)))
    x = np.random.uniform(0, 1)
    return (x < prob)

def metropolis(district, state, run, iter=1000):
    prior_ranges = get_prior_ranges()
    proposal_sigmas = get_proposal_sigmas(prior_ranges)
    _, df_train, _ = get_data(district, state)

    theta = param_init(prior_ranges)
    accepted = [theta]
    rejected = list()

    for i in tqdm(range(iter+1)):
        
        theta_new = proposal(theta, proposal_sigmas)
        if accept(theta, theta_new, df_train):
            theta = theta_new
        else:
            rejected.append(theta_new)
        accepted.append(theta)

        if i % 1000 == 0:
            new_name = './mcmc/{}_{}_run_{}_iter_{}.npy'.format(state, district, run, i)
            np.save(new_name, {"accepted":accepted, "rejected":rejected})
            if i != 0:
                old_name = './mcmc/{}_{}_run_{}_iter_{}.npy'.format(state, district, run, i - 1000)
                os.remove(old_name)
    
    return accepted, rejected

def get_PI(pred_dfs, date, key, multiplier=1.96):
    pred_samples = list()
    for df in pred_dfs:
        pred_samples.append(df.loc[date, key])
        
    mu = np.array(pred_samples).mean()
    sigma = np.array(pred_samples).std()
    low = mu - multiplier*sigma
    high = mu + multiplier*sigma
    return mu, low, high

def run_mcmc(district, state, n_chains=50, iters=100000):
    if not os.path.exists('./mcmc/'):
        os.makedirs('./mcmc/')
    mcmc = Parallel(n_jobs=mp.cpu_count())(delayed(metropolis)(district, state, run, iters) for run in range(n_chains))
    return mcmc

def visualize(mcmc, district, state):
    print("\nCalculating and plotting confidence intervals")
    df_district, df_train, df_val = get_data(district, state)
    data_split = df_district.copy()
    optimiser = Optimiser()
    default_params = optimiser.init_default_params(data_split)
    
    combined_acc = list()
    for k, run in enumerate(mcmc):
        burn_in = int(len(run) / 2)
        combined_acc += run[0][burn_in:]

    n_samples = 1000
    sample_indices = np.random.uniform(0, len(combined_acc), n_samples)

    pred_dfs = list()
    for i in tqdm(sample_indices):
        params = combined_acc[int(i)].copy()
        del params['sigma']
        pred_dfs.append(optimiser.solve(params, default_params, data_split))

    for df in pred_dfs:
        df.set_index('date', inplace=True)

    result = pred_dfs[0].copy()
    for col in result.columns:
        result["{}_low".format(col)] = ''
        result["{}_high".format(col)] = ''

    for date in tqdm(pred_dfs[0].index):
        for key in pred_dfs[0]:
            result.loc[date, key], result.loc[date, "{}_low".format(key)], result.loc[date, "{}_high".format(key)] = get_PI(pred_dfs, date, key)

    data_split.set_index("date", inplace=True)

    plt.figure(figsize=(15, 10))
    plt.plot(data_split['total_infected'].tolist(), c='g', label='Actual')
    plt.plot(result['total_infected'].tolist(), c='r', label='Estimated')
    plt.plot(result['total_infected_low'].tolist(), c='r', linestyle='dashdot')
    plt.plot(result['total_infected_high'].tolist(), c='r', linestyle='dashdot')
    plt.axvline(x=len(df_train), c='b', linestyle='dashed')
    plt.xlabel("Day")
    plt.ylabel("Total infected")
    plt.legend()
    plt.title(r"95% confidence intervals for {}, {}".format(district, state))
    
    plt.savefig('./mcmc_confidence_intervals_{}_{}.png'.format(district, state))

def analyse_runs(mcmc, district, state):
    print("\nPlotting all runs separately")
    plt.figure(figsize=(30, len(mcmc)*10))
    df_district, df_train, df_val = get_data(district, state)

    for k, run in enumerate(mcmc):
        data_split = df_district.copy()
        optimiser = Optimiser()
        default_params = optimiser.init_default_params(data_split)
        
        acc, rej = run[0], run[1]
        df_samples = pd.DataFrame(acc)
        
        plt.subplot(len(mcmc), 3, 3*k + 1)
        for param in df_samples.columns:
            plt.plot(list(range(len(df_samples[param]))), df_samples[param], label=param)
        plt.xlabel("iterations")
        plt.legend()
        plt.title("Accepted samples from run {}".format(k+1))
        
        rej_samples = pd.DataFrame(rej)
        
        plt.subplot(len(mcmc), 3, 3*k + 2)
        for param in rej_samples.columns:
            plt.scatter(list(range(len(rej_samples[param]))), rej_samples[param], label=param, s=2)
        plt.xlabel("iterations")
        plt.legend()
        plt.title("Rejected samples from run {}".format(k+1))
        
        burn_in = int(len(acc) / 2)
        n_samples = 1000
        posterior_samples = acc[burn_in:]
        sample_indices = np.random.uniform(0, len(posterior_samples), n_samples)

        pred_dfs = list()
        for i in tqdm(sample_indices):
            params = posterior_samples[int(i)].copy()
            del params['sigma']
            pred_dfs.append(optimiser.solve(params, default_params, data_split))
            
        for df in pred_dfs:
            df.set_index('date', inplace=True)
            
        result = pred_dfs[0].copy()
        for col in result.columns:
            result["{}_low".format(col)] = ''
            result["{}_high".format(col)] = ''
            
        for date in tqdm(pred_dfs[0].index):
            for key in pred_dfs[0]:
                result.loc[date, key], result.loc[date, "{}_low".format(key)], result.loc[date, "{}_high".format(key)] = get_PI(pred_dfs, date, key)
                
        data_split.set_index("date", inplace=True)

        plt.subplot(len(mcmc), 3, 3*k + 3)
        plt.plot(data_split['total_infected'].tolist(), c='g', label='Actual')
        plt.plot(result['total_infected'].tolist(), c='r', label='Estimated')
        plt.plot(result['total_infected_low'].tolist(), c='r', linestyle='dashdot')
        plt.plot(result['total_infected_high'].tolist(), c='r', linestyle='dashdot')
        plt.axvline(x=len(df_train), c='b', linestyle='dashed')
        plt.xlabel("Day")
        plt.ylabel("Total infected")
        plt.legend()
        plt.title("95% CIs for {}, {} from run {}".format(district, state, k+1))
        
    plt.savefig("./mcmc_runs_{}_{}.png".format(district, state))

def main():
    regions = [('Delhi', ''), ('Karnataka', 'Bengaluru'), ('Maharashtra', 'Mumbai'), ('Maharashtra', 'Pune'), ('Gujarat', 'Ahmadabad'), ('Rajasthan', 'Jaipur')]
    state, district = regions[0]

    mcmc = run_mcmc(district, state, n_chains=50, iters=100000)
    visualize(mcmc, district, state)
    analyse_runs(mcmc, district, state)
    import pdb; pdb.set_trace()

if __name__=='__main__':
    main()


####################################################
# ## Checking validity with Gelman-Rubin statistics

# Check Section 4.2 [here](http://www.columbia.edu/~mh2078/MachineLearningORFE/MCMC_Bayes.pdf)
####################################################


def accumulate(dict_list):
    accumulator = defaultdict(int)
    for elt in dict_list:
        for key in elt:
            accumulator[key]+=elt[key]
    return accumulator

def divide(dictvar, num):
    return {key:dictvar[key]/num for key in dictvar}

            
def avg_sum_chain(chain):
    chain_sums_avg = accumulate(chain)
    return divide(chain_sums_avg, len(chain))

def avg_sum_multiple_chains(chain_sums_avg):
    multiple_chain_sums_avg = accumulate(chain_sums_avg)
    return divide(multiple_chain_sums_avg, len(chain_sums_avg))

def compute_B(multiple_chain_sums_avg, chain_sums_avg, n, m):
    B = defaultdict(int)
    for elt in chain_sums_avg:
        for key in elt:
            B[key] += np.square(elt[key] - multiple_chain_sums_avg[key])
    return divide(B, (m-1)/n)

def compute_W(split_chains, chain_sums_avg, n, m):
    s = []
    for j in range(m):
        s_j_sq = defaultdict(int)
        chain = split_chains[j]
        chain_sum_avg_j = chain_sums_avg[j]
        for i in range(n):
            chain_elt = chain[i]
            for key in chain_elt:
                s_j_sq[key] += np.square(chain_elt[key] - chain_sum_avg_j [key])
        s_j_sq = divide(s_j_sq, n - 1)
        s.append(s_j_sq)
    return (divide (accumulate(s),m))

def divide_dict(d1, d2):
    accumulator = defaultdict(int)
    for key in d1:
        accumulator[key] = d1[key]/d2[key]
    return accumulator


# # In[21]:


# burn_in = int(len(mcmc[0][0]) / 2)
# chains = [mcmc_chain[0] for mcmc_chain in mcmc]
# burn_in = int(len(chains[0]) / 2)
# sampled_chains = [chain[:burn_in] for chain in chains]
# split_chains = [sampled_chain[int(burn_in/2):] for sampled_chain in sampled_chains]             + [sampled_chain[:int(burn_in/2)] for sampled_chain in sampled_chains]


# # In[22]:


# chain_sums_avg = []
# for chain in split_chains:
#     chain_sums_avg.append(avg_sum_chain(chain))
# multiple_chain_sums_avg = avg_sum_multiple_chains(chain_sums_avg) 


# # In[24]:


# multiple_chain_sums_avg


# # In[56]:


# m = len(split_chains)
# n = len(split_chains[0])
# W =  compute_W(split_chains, chain_sums_avg, n, m)
# B =  compute_B(multiple_chain_sums_avg, chain_sums_avg, n, m)
# var_hat = accumulate([divide(W, n/(n-1)), divide(B, n) ])
# R_hat_sq = divide_dict(var_hat, W)
# R_hat = {key:np.sqrt(value) for key, value in R_hat_sq.items()}
# neff = divide_dict(var_hat, B)
# neff = {key: m*n*value for key, value in neff.items()}


# # In[57]:


# neff


# # In[53]:


# R_hat

