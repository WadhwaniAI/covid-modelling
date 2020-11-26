import numpy as np
from collections import OrderedDict
def params_from_trials(trials):
    selected_ACC = []
    T = trials.trials
    param_keys = list(T[0]['misc']['vals'].keys())

    for i in range(len(T)):
        a = list(T[i]['misc']['vals'].values())
        a = [i[0] for i in a]
        selected_ACC.append(OrderedDict(zip(param_keys,a)))
    return selected_ACC

def calc_DIC(selected_ACC,df_train,_default_params,_optimiser,compartments):

    selected_LIK = [_log_likelihood(i,df_train,_default_params,_optimiser,compartments) for i in selected_ACC]
    deviance = np.array(selected_LIK)*-2
    mean_deviance = np.mean(deviance)
    theta_dict_keys = list(selected_ACC[0].keys())
    acc = [list(i.values()) for i in selected_ACC]
    acc = np.array(acc)
    m_acc = np.mean(acc,axis =0)
    m_acc_sample = OrderedDict(zip(theta_dict_keys,m_acc))
    deviance_at_mean = _log_likelihood(m_acc_sample,df_train,_default_params,_optimiser,compartments)*-2
    return 2 * mean_deviance - deviance_at_mean 





def _gaussian_log_likelihood( true, pred, sigma):
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
    ll = ll/(N**2)
    return ll

def _log_likelihood(theta,df_train,_default_params,_optimiser,compartments):
    """
    Computes and returns the data log-likelihood as per either a gaussian or poisson distribution.
    
    Args:
        theta (OrderedDict): parameter-value pairs.
    
    Returns:
        float: log-likelihood of the data.
    """
    ll = 0
    params_dict = {**theta, **_default_params}
    df_prediction = _optimiser.solve(params_dict,end_date = df_train[-1:]['date'].item())
    sigma = 250


    for compartment in compartments:
        pred = np.array(df_prediction[compartment], dtype=np.int64)
        true = np.array(df_train[compartment], dtype=np.int64)
        pred = np.ediff1d(pred)
        true = np.ediff1d(true)
        ll += _gaussian_log_likelihood(true, pred, sigma)

    return ll