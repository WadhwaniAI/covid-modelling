"""Summary
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_state(district):
    """Summary
    
    Args:
        district (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    state_map = {
        "Mumbai": "Maharashtra",
        "Pune": "Maharashtra",
        "Jaipur": "Rajasthan",
        "Bengaluru Urban": "Karnataka",
        "Bengaluru Rural": "Karnataka",
        "Ahmedabad": "Gujarat",
        "North Delhi": "Delhi",
        "South Delhi": "Delhi",
        "East Delhi": "Delhi",
        "West Delhi": "Delhi"
    }
    return state_map[district]
    
def get_PI(pred_dfs, date, key, multiplier=1.96):
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

def set_optimizer(data: pd.DataFrame, train_period: int,default_params, optimiser):
    """Summary
    
    Args:
        data (pd.DataFrame): Description
        train_period (int): Description
    
    Returns:
        TYPE: Description
    """
    #optimiser = Optimiser()
    default_params = optimiser.init_default_params(data,default_params,train_period=train_period)
    return optimiser, default_params

def predict(data: pd.DataFrame, mcmc_runs: list, default_params: dict, optimiser: object,
      predict_days: int, end_date: str = None) -> pd.DataFrame:
    """Summary
    
    Args:
        data (pd.DataFrame): Description
        mcmc_runs (list): Description
        predict_days (int): Description
        end_date (str, optional): Description
    
    Returns:
        pd.DataFrame: Description
    """
    #optimiser, default_params = set_optimizer(data, predict_days, default_params)
    
    combined_acc = list()
    for _, run in enumerate(mcmc_runs):
        burn_in = int(len(run) / 2)
        combined_acc += run[0][burn_in:]
        
    n_samples = 1000
    sample_indices = np.random.uniform(0, len(combined_acc), n_samples)
    
    pred_dfs = list()
    for i in tqdm(sample_indices):
        #import pdb; pdb.set_trace()
        all_params = {**combined_acc[int(i)], **default_params}
        pred_dfs.append(optimiser.solve(params_dict = all_params, end_date=end_date))
    for df in pred_dfs:
        df.set_index('date', inplace=True)

    result = pred_dfs[0].copy()
    for col in result.columns:
        result["{}_low".format(col)] = ''
        result["{}_high".format(col)] = ''

    for date in tqdm(pred_dfs[0].index):
        for key in pred_dfs[0]:
            result.loc[date, key], result.loc[date, "{}_low".format(key)], result.loc[date, "{}_high".format(key)] = get_PI(pred_dfs, date, key)
    
    return result

def accumulate(dict_list):
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

def divide(dictvar, num):
    """Summary
    
    Args:
        dictvar (TYPE): Description
        num (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return {key:dictvar[key]/num for key in dictvar}
            
def avg_sum_chain(chain):
    """Summary
    
    Args:
        chain (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    chain_sums_avg = accumulate(chain)
    return divide(chain_sums_avg, len(chain))

def avg_sum_multiple_chains(chain_sums_avg):
    """Summary
    
    Args:
        chain_sums_avg (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    multiple_chain_sums_avg = accumulate(chain_sums_avg)
    return divide(multiple_chain_sums_avg, len(chain_sums_avg))

def compute_B(multiple_chain_sums_avg, chain_sums_avg, n, m):
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
    return divide(B, (m-1)/n)

def compute_W(split_chains, chain_sums_avg, n, m):
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
        s_j_sq = divide(s_j_sq, n - 1)
        s.append(s_j_sq)
    return (divide (accumulate(s),m))

def divide_dict(d1, d2):
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

def get_formatted_trials(params, losses):
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


