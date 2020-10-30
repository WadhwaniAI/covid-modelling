import matplotlib.pyplot as plt
import pandas as pd
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

    df_sorted = df.sort_values(var_name)
    annot_mat = df_sorted.index.to_numpy().reshape(-1, 3) + " : " + \
    np.around(df_sorted[var_name].to_numpy().reshape(-1, 3), 2).astype(str)
    annot_mat = annot_mat.astype(str).tolist()
    sns.heatmap(df_sorted[var_name].to_numpy().reshape(-1, 3), 
                cmap='coolwarm', center=center, xticklabels=False,
                yticklabels=False, annot=annot_mat, fmt='', ax=ax)

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
