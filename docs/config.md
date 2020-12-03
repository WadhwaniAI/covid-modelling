[This](../configs/seir/default.yaml) is what a default config looks like. 

It has 4 top level keys :

- `fitting` : everything related to fitting
- `sensitivity` : parameters related to the sensitivity of the fit
- `forecast` : parameters related to forecasting
- `uncertainty` : parameters related to uncertainty estimation

We may add a 5th key for `whatifs` scenarios, but that depends on how formally intergrated that becomes to the end to end pipeline.

## Fitting

- `data` :
- `model` : 
- `variable_param_ranges` :
- `default_params` :
- `fitting_method` :
- `fitting_method_params` : str. Can be `gridsearch` or  `bayes_opt`
    - `num_evals`: int
    - `algo`: str. Can be `tpe` (TPE), `atpe` (adaptive TPE), or `rand` (random search)
    - `seed`: int
- `split` :
    - `start_date`: date. Start of training period. Default null
    - `end_date`: date. End of training period. Default null
    - `train_period`: int. Length of training period.
    - `val_period`: int. Length of validation period.
    - `test_period`: int. Length of test period.
Giving train, val and test period is a must. If both start and end date are null, it fits on latest data. Both can't be given, only 1 of them can be given. Otherwise an error will be thrown.
- `loss` :
    - `loss_method` : str. Can be `rmse`, `mape`, or `rmsle`
    - `loss_compartments` : \[str\]
    - `loss_weights` : \[float\]

## Forecast

- `forecast_days` : int. 
- `num_trials_to_plot` : int. Num trials to plot in the plot_topk_trials functions
- `plot_topk_trials_for_columns` : \[str\]. Which columns to plot topk trials for. 
- `plot_ptiles_for_columns` :\[str\]. Which columns to plot percentiles for. 

## Uncertainty

- `method`: MCUncertainty, or MCMCUncertainty

Parameters for each uncertainty class come inside `uncertainty_params`. 

For MCUncertainty : 
- `num_evals` : int
- `fitting_method`: str. Can be `gridsearch` or  `bayes_opt`
For `gridsearch` -  
- `fitting_method_params`:
    - `parallelise`: bool
Specify the range for beta - 
- `variable_param_ranges`:
    - `beta`: [[0, 10], 101] (Eg)
- `construct_percentiles_day_wise`: bool. If true, percentiles are constructed day wise.
- `date_of_sorting_trials` : date
- `sort_trials_by_column` : str
- `loss` :
    - `loss_method` : str. Can be `rmse`, `mape`, or `rmsle`
    - `loss_compartments` : \[str\]
    - `loss_weights` : \[float\]
- `percentiles`: \[float\]. Which percentiles to compute. 