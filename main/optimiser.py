import numpy as np
import pandas as pd
from main.losses import Loss_Calculator

class Optimiser():

    def __init__(self):
        self.loss_calculator = Loss_Calculator()

    def solve_and_compute_loss(variable_params, default_params, df_total, total_days, train_indices=[-20, -10], loss_method='rmse'):
        params_dict = {**variable_params, **default_params}
        vanilla_params, testing_params, state_init_values = init_params(**params_dict)
        solver = SEIR_Testing(vanilla_params, testing_params, state_init_values)
        sol = solver.solve_ode(total_no_of_days=total_days - 1, time_step=1, method='Radau')
        states_time_matrix = (sol.y*vanilla_params['N']).astype('int')

        loss = self.loss_calculator.calc_loss(states_time_matrix[:, train_indices[0]:train_indices[1]], \ 
        df.iloc[train_indices[0]:train_indices[1], :], loss_method)
        return loss

    def _create_dict(param_names, values):
        params_dict = {param_names[i]: values[i] for i in range(len(values))}
        return params_dict

    def init_optimiser(df_true, N=1e7, lockdown_date='2020-03-25', lockdown_removal_date='2020-05-03', T_hosp = 0.001, P_fatal = 0.01):
        init_infected = max(df_true.loc[0, 'Total Infected'], 1)
        start_date = df_true.loc[0, 'Date']
        intervention_date = datetime.datetime.strptime(lockdown_date, '%Y-%m-%d')
        lockdown_removal_date = datetime.datetime.strptime(lockdown_removal_date, '%Y-%m-%d')

        default_params = {
            'N' : N,
            'init_infected' : init_infected,
            'intervention_day' : (intervention_date - start_date).days,
            'lockdown_removal_day' : (lockdown_removal_date - start_date).days,
            'T_hosp' : T_hosp,
            'P_fatal' : P_fatal
        }
        return default_params

    def gridsearch(df_true, default_params, variable_param_ranges, method='rmse'):
        
        total_days = len(df_true['Date'])

        variable_param_ranges = {
            'R0' : np.linspace(1.8, 3, 13),
            'T_inc' : np.linspace(3, 5, 5),
            'T_inf' : np.linspace(2.5, 3.5, 5),
            'T_recov_severe' : np.linspace(11, 15, 10),
            'P_severe' : np.linspace(0.3, 0.9, 25),
            'intervention_amount' : np.linspace(0.4, 1, 31)
        }
        
        rangelists = list(variable_param_ranges.values())
        cartesian_product_tuples = itertools.product(*rangelists)
        list_of_param_dicts = [self._create_dict(list(variable_param_ranges.keys()), values) for values in cartesian_product_tuples]

        partial_solve_and_compute_loss = partial(solve_and_compute_loss, default_params = default_params, df=df_true_fitting, 
                                                 total_days=total_days, method = method) 
        
        loss_array = Parallel(n_jobs=40)(delayed(partial_solve_and_compute_loss)(params_dict) for params_dict in tqdm(list_of_param_dicts))
                    
        return loss_array, list_of_param_dicts, df_true

    def bayes_opt(df_true, default_params, variable_param_ranges, method='rmse', num_evals=3500):
        
        variable_params = {
            'R0' : hp.uniform('R0', 1.6, 3),
            'T_inc' : hp.uniform('T_inc', 4, 5),
            'T_inf' : hp.uniform('T_inf', 3, 4),
            'T_recov_severe' : hp.uniform('T_recov_severe', 9, 20),
            'P_severe' : hp.uniform('P_severe', 0.3, 0.99),
            'intervention_amount' : hp.uniform('intervention_amount', 0.3, 1)
        }
        
        partial_solve_and_compute_loss = partial(solve_and_compute_loss, default_params = default_params, df=df_true_fitting, 
                                                 total_days=total_days, method = method) 
        
        searchspace = variable_param_ranges
        
        trials = Trials()
        best = fmin(partial_solve_and_compute_loss,
                    space=searchspace,
                    algo=tpe.suggest,
                    max_evals=num_evals,
                    trials=trials)
        
        return best, trials

    def evaluate_losses(best, default_params, df_train, df_val):
        start_date = df_train.iloc[0, 0]
        simulate_till = df_val.iloc[-1, 0]
        total_no_of_days = (simulate_till - start_date).days + 1
        no_of_train_days = (df_train.iloc[-1, 0] - start_date).days + 1
        no_of_val_days = total_no_of_days - no_of_train_days
        
        final_params = {**best, **default_params}
        vanilla_params, testing_params, state_init_values = init_params(**final_params)
        solver = SEIR_Testing(vanilla_params, testing_params, state_init_values)
        sol = solver.solve_ode(total_no_of_days=total_no_of_days - 1, time_step=1, method='Radau')
        states_time_matrix = (sol.y*vanilla_params['N']).astype('int')

        train_output = states_time_matrix[:, :no_of_train_days]
        val_output = states_time_matrix[:, -no_of_val_days:]
        
        rmse_loss = calc_loss_dict(train_output, df_train, method='rmse')
        rmse_loss = pd.DataFrame.from_dict(rmse_loss, orient='index', columns=['rmse'])
        
        mape_loss = calc_loss_dict(train_output, df_train, method='mape')
        mape_loss = pd.DataFrame.from_dict(mape_loss, orient='index', columns=['mape'])
        
        train_losses = pd.concat([rmse_loss, mape_loss], axis=1)
        
        pred_hospitalisations = val_output[6] + val_output[7] + val_output[8]
        pred_recoveries = val_output[9]
        pred_fatalities = val_output[10]
        pred_infectious_unknown = val_output[2] + val_output[4]
        pred_total_cases = pred_hospitalisations + pred_recoveries + pred_fatalities
        print('Pred', pred_hospitalisations, pred_total_cases)
        print('True', df_val['Hospitalised'].to_numpy(), df_val['Total Infected'].to_numpy())
        
        rmse_loss = calc_loss_dict(val_output, df_val, method='rmse')
        rmse_loss = pd.DataFrame.from_dict(rmse_loss, orient='index', columns=['rmse'])
        
        mape_loss = calc_loss_dict(val_output, df_val, method='mape')
        mape_loss = pd.DataFrame.from_dict(mape_loss, orient='index', columns=['mape'])
        
        val_losses = pd.concat([rmse_loss, mape_loss], axis=1)
        
        print(train_losses)
        print(val_losses)
        
        return train_losses, val_losses
