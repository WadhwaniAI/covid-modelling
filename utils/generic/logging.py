import wandb
import copy
import matplotlib as mpl

def log_wandb(predictions_dict):
    # Logging all input data dataframes
    dataframes_to_log = ['df_train', 'df_val', 'df_district', 'df_district_unsmoothed']
    for run in ['m1', 'm2']:
        for df_name in dataframes_to_log:
            df = copy.deepcopy(predictions_dict[run][df_name])
            if df is not None:
                df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            wandb.log({f"{run}/{df_name}": wandb.Table(dataframe=df)})
        wandb.log({f"{run}/{'df_loss'}": wandb.Table(dataframe=predictions_dict[run]['df_loss'])})

    # Logging all forecast data dataframes
    for forecast_name in predictions_dict['m2']['forecasts'].keys():
        df = copy.deepcopy(predictions_dict['m2']['forecasts'][forecast_name])
        if df is not None:
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        wandb.log({f"{'forecast_dataframes'}/{forecast_name}": wandb.Table(dataframe=df)})


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
