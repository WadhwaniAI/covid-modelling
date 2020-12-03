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
- `fitting_method_params` :
- `split` :
- `loss` :

## Forecast

- `forecast_days` :
- `num_trials_to_plot` :
- `plot_topk_trials_for_columns` :
- `plot_ptiles_for_columns` :

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
    - `loss_method` : str
    - `loss_compartments` : \[str\]
    - `loss_weights` : \[float\]