import matplotlib.pyplot as plt
import geoplot as gplt
import geoplot.crs as gcrs

import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns


def calculate_z_score(df_mape, df_rank, model_name='Wadhwani_AI'):
    df = pd.concat([df_mape.mean(axis=0), df_mape.std(axis=0), 
                    df_mape.median(axis=0), df_mape.mad(axis=0),
                    df_mape.loc[model_name, :], df_rank.loc[model_name, :]], axis=1)
    df.columns = ['mean_mape', 'std_mape', 'median_mape',
                  'mad_mape', 'model_mape', 'model_rank']
    df['z_score'] = (df['model_mape'] - df['mean_mape'])/(df['std_mape'])
    df['non_param_z_score'] = np.abs(df['model_mape'] - df['median_mape'])/(df['mad_mape'])
    return df


def create_heatmap(df, var_name='z_score', center=0):
    fig, ax = plt.subplots(figsize=(12, 12))

    df_sorted = df.sort_values('z_score')
    annot_mat = df_sorted.index.to_numpy().reshape(-1, 3) + " : " + \
    np.around(df_sorted[var_name].to_numpy().reshape(-1, 3), 2).astype(str)
    annot_mat = annot_mat.astype(str).tolist()
    sns.heatmap(df_sorted[var_name].to_numpy().reshape(-1, 3), 
                cmap='coolwarm', center=center, xticklabels=False,
                yticklabels=False, annot=annot_mat, fmt='', ax=ax)

    ax.set_title(f'Heatmap of {var_name} for all US states, sorted by z_score, cmap centered at {center}')
    return fig, ax


def create_geoplot_choropleth(df):
    contiguous_usa = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))
    fig, ax = plt.subplots(figsize=(12, 8))
    gplt.choropleth(
        contiguous_usa.merge(df, left_on='state', right_index=True),
        hue='z_score', projection=gcrs.AlbersEqualArea(),
        edgecolor='black', linewidth=1,
        cmap='coolwarm', legend=True,
        ax=ax
    )
    return fig

def combine_with_train_error(predictions_dict, df):
    df_wadhwani = pd.DataFrame(index=list(predictions_dict.keys()),
                               columns=['best_loss_train', 'test_loss', 
                                        'T_recov_fatal', 'P_fatal'])
    for loc in predictions_dict.keys():
        df_wadhwani.loc[loc, 'best_loss_train'] = predictions_dict[loc]['m2']['df_loss'].to_numpy()[0][0]
        df_wadhwani.loc[loc, 'T_recov_fatal'] = predictions_dict[loc]['m2']['best_params']['T_recov_fatal']
        df_wadhwani.loc[loc, 'P_fatal'] = predictions_dict[loc]['m2']['best_params']['P_fatal']

    df_wadhwani = df_wadhwani.merge(df, left_index=True, right_index=True)

    df_wadhwani.drop(['Northern Mariana Islands', 'Guam',
                      'Virgin Islands'], axis=0, inplace=True, errors='ignore')
    
    return df_wadhwani

def create_scatter_plot_mape(df_wadhwani, annotate=True, abbv=False, abbv_dict=None):
    fig, ax = plt.subplots(figsize=(12, 12))
    df_zpos = df_wadhwani[df_wadhwani['z_score'] > 0]
    df_zneg = df_wadhwani[df_wadhwani['z_score'] < 0]
    ax.scatter(df_zpos['best_loss_train'], df_zpos['model_mape'],
            s=df_zpos['z_score']*50, c='red', marker='o', label='+ve Z score states')
    ax.scatter(df_zneg['best_loss_train'], df_zneg['model_mape'], s=-
            df_zneg['z_score']*50, c='blue', marker='o', label='-ve Z score states')
    if annotate:
        for i, (index, row) in enumerate(df_wadhwani.iterrows()):
            if abbv:
                state_name = abbv_dict[index]
            else:
                state_name = index
            ax.annotate(f'{state_name} ({round(row["z_score"], 2)})',
                        (row['best_loss_train'], row['model_mape']))
    ax.set_xlabel('MAPE on training data (calculated daily)')
    ax.set_ylabel('MAPE on unseen data (calculated weekly)')
    ax.axvline(1, ls=':', c='red', label='train error threshold (1%)')
    ax.axhline(5, ls=':', c='blue', label='test error threshold (5%)')
    ax.legend()
    ax.set_title(f'Scatter plot of train vs test error, point radii proportional to z_score')
    return fig
