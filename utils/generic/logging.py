import copy
import json
import numbers
import os

import matplotlib as mpl
import mlflow
import numpy as np
import pandas as pd
import wandb


def create_dataframes_for_logging(m2_dict):
    """Function for creating dataframes of top k trials params, deciles params and deciles losses
    Takes as input the dict of m2 fits, and returns 3 pd.DataFrames

    Args:
        m2_dict (dict): predictions_dict['m2'] as returned by generated_report.ipynb

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame: dataframes as described above
    """
    trials_processed = copy.deepcopy(m2_dict['trials_processed'])
    trials_processed['losses'] = np.around(trials_processed['losses'], 2)
    trials_processed['params'] = [{key: np.around(value, 2) for key, value in params_dict.items()}
                                   for params_dict in trials_processed['params']]
    df = pd.DataFrame.from_dict({(i+1, trials_processed['losses'][i]): trials_processed['params'][i]
                                 for i in range(10)})
    df_trials_params = copy.deepcopy(df)

    deciles = copy.deepcopy(m2_dict['deciles'])
    for key in deciles.keys():
        deciles[key]['params'] = {param: np.around(value, 2) 
                                  for param, value in deciles[key]['params'].items()}

    for key in deciles.keys():
        deciles[key]['df_loss'] = deciles[key]['df_loss'].astype(
            float).round(2)

    df = pd.DataFrame.from_dict({np.around(key, 1): deciles[key]['params'] 
                                 for key in deciles.keys()})
    df_deciles_params = copy.deepcopy(df)

    df = pd.DataFrame.from_dict({np.around(key, 1): deciles[key]['df_loss'].to_dict()['train']
                                 for key in deciles.keys()})
    df_decile_losses = copy.deepcopy(df)

    return df_trials_params, df_deciles_params, df_decile_losses


def log_wandb(predictions_dict):
    """Function for logging stuff to W&B. Logs config param during initialisation,
       matplotlib plots, and pandas dataframes

    Args:
        predictions_dict (dict): Dict of all predictions generated in generate_report.ipynb
    """
    # Logging all input data dataframes
    dataframes_to_log = ['df_train', 'df_val', 'df_district', 'df_district_unsmoothed']
    for run in ['m1', 'm2']:
        for df_name in dataframes_to_log:
            df = copy.deepcopy(predictions_dict[run][df_name])
            if df is not None:
                df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            wandb.log({f"{run}/{df_name}": wandb.Table(dataframe=df)})
        wandb.log({f"{run}/{'df_loss'}": wandb.Table(
            dataframe=predictions_dict[run]['df_loss'].reset_index())})

    # Logging all forecast data dataframes
    for forecast_name in predictions_dict['m2']['forecasts'].keys():
        df = copy.deepcopy(predictions_dict['m2']['forecasts'][forecast_name])
        if df is not None:
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        wandb.log({f"{'forecast_dataframes'}/{forecast_name}": wandb.Table(dataframe=df)})

    df_trials_params, df_deciles_params, df_decile_losses = create_dataframes_for_logging(
        predictions_dict['m2'])

    wandb.log({f"{'deciles'}/{'trials_params'}": wandb.Table(
        dataframe=df_trials_params.reset_index())})
    wandb.log({f"{'deciles'}/{'deciles_params'}": wandb.Table(
        dataframe=df_deciles_params.reset_index())})
    wandb.log({f"{'deciles'}/{'decile_losses'}": wandb.Table(
        dataframe=df_decile_losses.reset_index())})

    # Logging all non forecast plots
    for run in ['m1', 'm2']:
        for key, value in predictions_dict[run]['plots'].items():
            if type(value) == mpl.figure.Figure and 'forecast' not in key:
                wandb.log({f"{run}/{key}": [wandb.Image(value)]})

    # Logging all forecast plots
    for key, value in predictions_dict['m2']['plots'].items():
        if type(value) == mpl.figure.Figure and 'forecast' in key:
            wandb.log({f"{'forecasts'}/{key}": [wandb.Image(value)]})
    
    for key, value in predictions_dict['m2']['plots']['forecasts_topk'].items():
        wandb.log({f"{'forecasts'}/forecasts_topk_{key}": [wandb.Image(value)]})

    for key, value in predictions_dict['m2']['plots']['forecasts_ptiles'].items():
        wandb.log({f"{'forecasts'}/forecasts_ptiles_{key}": [wandb.Image(value)]})


def log_mlflow(experiment_name='covid-modelling-default', run_name=None, params=None, metrics=None, artifact_dir=None,
               tags=None):
    """Function for logging to MLflow

    Args:
        experiment_name (str): name of MLflow experiment
        run_name (str): name of the run
        params (dict): dictionary of parameters to log
        metrics (dict): dictionary of metrics to log
        artifact_dir (str): path containing artifacts to log
        tags (dict): dictionary of tags to log
    """
    mlflow.set_experiment(experiment_name)
    cred_file_path = '../../misc/mlflow/credentials.json'
    if not os.path.isfile(cred_file_path):
        print('MLflow credentials not found')
    else:
        with open(cred_file_path, 'r') as infile:
            mlflow_config = json.load(infile)
        os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_config['username']
        os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_config['password']
        mlflow.set_tracking_uri(mlflow_config['tracking_url'])

        with mlflow.start_run(run_name=run_name):
            if params is not None:
                print('Logging parameters...')
                mlflow.log_params(params)
            if metrics is not None:
                print('Logging metrics...')
                for metric, value in metrics.items():
                    if isinstance(value, numbers.Number):
                        mlflow.log_metric(key=metric, value=value)
            if artifact_dir is not None:
                print(f'Logging artifacts from local directory: {artifact_dir}')
                mlflow.log_artifacts(artifact_dir)
            if tags is not None:
                print('Logging tags...')
                mlflow.set_tags(tags)
