[This](../configs/seir/default.yaml) is what a default config looks like. 

It has 4 top level keys :

- `fitting` : everything related to fitting
- `sensitivity` : parameters related to the sensitivity of the fit
- `forecast` : parameters related to forecasting
- `uncertainty` : parameters related to uncertainty estimation

We may add a 5th key for `whatifs` scenarios, but that depends on how formally intergrated that becomes to the end to end pipeline.

## Fitting

- `data` :
    - `tables` : list of tables to retrieve from Athena DB
    - `schema_name` : name of the schema in Athena DB
    - `staging_dir` : name of the S3 staging directory
    - `label` : label used to name all associated artifacts 
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
- `num_evals` : 
- `date_of_sorting_trials` : 
- `sort_trials_by_column` : 
- `loss` :
    - `loss_method` : 
    - `loss_compartments` : 
    - `loss_weights` : 