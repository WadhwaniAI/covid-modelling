"""Summary
"""
from collections import defaultdict

import numpy as np

    
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


