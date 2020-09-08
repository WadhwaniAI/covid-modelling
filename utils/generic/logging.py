import wandb
import matplotlib as mpl
wandb.init(project="covid-modelling")

def log_wandb(predictions_dict):
    for key, value in predictions_dict['m2']['plots'].items():
        if type(value) == mpl.figure.Figure:
            wandb.log({f"{'m2'}/{key}": [wandb.Image(value)]})
    
    for key, value in predictions_dict['m2']['plots']['forecasts_topk'].items():
        wandb.log({f"{'m2'}/forecasts_topk_{key}": [wandb.Image(value)]})

    for key, value in predictions_dict['m2']['plots']['forecasts_ptiles'].items():
        wandb.log({f"{'m2'}/forecasts_ptiles_{key}": [wandb.Image(value)]})

    wandb.log({f"{'m2'}/{'ground_truth'}": [wandb.Table(dataframe=predictions_dict['m2']['df_district'])] })
    wandb.log({f"{'m2'}/prediction": [wandb.Table(dataframe=predictions_dict['m2']['df_prediction'])] })
