from copy import copy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.patches import Patch
import geoplot as gplt
import geoplot.crs as gcrs
from shapely.geometry import MultiPolygon

import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
import scipy.stats as stats
from scipy.stats import median_abs_deviation
import statsmodels.api as sm
import adjustText as aT

from viz.utils import add_inset_subplot_to_axes

def calculate_z_score(df_mape, df_rank, model_name='Wadhwani_AI'):
    """Function for calculating Z score and non param Z score

    Args:
        df_mape (pd.DataFrame): dataframes of mape values for all models, locations
        df_rank (pd.DataFrame): dataframes of ranks values for all models, locations
        model_name (str, optional): Which model to calculate Z scores for. Defaults to 'Wadhwani_AI'.

    Returns:
        pd.DataFrame: dataframe with the calculated Z scores
    """

    df = pd.concat([df_mape.mean(axis=0), df_mape.std(axis=0), 
                    df_mape.median(axis=0), df_mape.mad(axis=0),
                    df_mape.loc[model_name, :], df_rank.loc[model_name, :]], axis=1)
    df.columns = ['mean_mape', 'std_mape', 'median_mape',
                  'mad_mape', 'model_mape', 'model_rank']
    df['z_score'] = (df['model_mape'] - df['mean_mape'])/(df['std_mape'])
    df['non_param_z_score'] = (df['model_mape'] - df['median_mape'])/df['mad_mape']
    return df


def combine_with_train_error(predictions_dict, df):
    """Combine the Z score dataframe with the train error datafrom (read from predictions_dict file)

    Args:
        predictions_dict (dict): The predictions_dict output file
        df (pd.DataFrame): Z Score dataframe

    Returns:
        pd.DataFrame: df with the z scores and train error
    """
    df_wadhwani = pd.DataFrame(index=list(predictions_dict.keys()),
                               columns=['best_loss_train', 'test_loss',
                                        'T_recov_fatal', 'P_fatal'])
    for loc in predictions_dict.keys():
        df_wadhwani.loc[loc, 'best_loss_train'] = predictions_dict[loc]['m2']['df_loss'].to_numpy()[
            0][0]
        df_wadhwani.loc[loc,
                        'T_recov_fatal'] = predictions_dict[loc]['m2']['best_params']['T_recov_fatal']
        df_wadhwani.loc[loc,
                        'P_fatal'] = predictions_dict[loc]['m2']['best_params']['P_fatal']

    df_wadhwani = df_wadhwani.merge(df, left_index=True, right_index=True)

    df_wadhwani.drop(['Northern Mariana Islands', 'Guam',
                      'Virgin Islands'], axis=0, inplace=True, errors='ignore')

    return df_wadhwani

def create_performance_table(df_mape, df_rank):
    """Creates dataframe of all models sorted by MAPE value and rank, to compare relative performance 

    Args:
        df_mape (pd.DataFrame): df of MAPE values for all models, regions
        df_rank (pd.DataFrame): df of ranks for all models, regions

    Returns:
        pd.DataFrame: The performance talble
    """
    median_mape = df_mape.loc[:, np.logical_not(
        df_mape.loc['Wadhwani_AI-BayesOpt', :].isna())].median(axis=1).rename('median_mape')
    median_rank = df_rank.loc[:, np.logical_not(
        df_rank.loc['Wadhwani_AI-BayesOpt', :].isna())].median(axis=1).rename('median_rank')
    merged = pd.concat([median_mape, median_rank], axis=1)
    merged.reset_index(inplace=True)
    merged['model1'] = merged['model']
    merged = merged[['model', 'median_mape', 'model1', 'median_rank']]
    merged = merged.sort_values('median_mape').round(2)
    merged.reset_index(drop=True, inplace=True)
    temp = copy(merged.loc[:, ['model1', 'median_rank']])
    temp.sort_values('median_rank', inplace=True)
    temp.reset_index(drop=True, inplace=True)
    merged.loc[:, ['model1', 'median_rank']
            ] = temp.loc[:, ['model1', 'median_rank']]
    merged.index = merged.index + 1
    return merged


def preprocess_shape_file(filename='cb_2018_us_state_5m/cb_2018_us_state_5m.shp', 
                          root_dir='../../data/data/shapefiles'):
    """Helper function for preprocessing shape file of US states

    Args:
        filename (str, optional): Shapefile filename. Defaults to 'cb_2018_us_state_5m/cb_2018_us_state_5m.shp'.
        root_dir (str, optional): Directory where shapefiles are stored. Defaults to '../../data/data/shapefiles'.
    
    Returns:
        gpd.GeoDataFrame: Returns the preprocessed geodataframe
    """
    gdf = gpd.read_file(f'{root_dir}/{filename}')
    gdf.rename({'NAME' : 'state'}, axis=1, inplace=True)
    # Removing territories
    states_to_remove = ['United States Virgin Islands', 'Puerto Rico', 'American Samoa', 
                        'Commonwealth of the Northern Mariana Islands', 'Guam']
    gdf = gdf[np.logical_not(gdf['state'].isin(states_to_remove))]

    # Pruning parts of Alaska where the Latitude is negative (ie, beyond -180)
    multi_polys = gdf.loc[gdf['state'] == 'Alaska', 'geometry'].to_numpy()[0]
    indices_to_keep = []
    for i, poly in enumerate(multi_polys):
        xcords = np.array(list(poly.exterior.coords))[:, 0]
        if sum(xcords > 0) != len(xcords):
            indices_to_keep.append(i)

    new_multi_polys = MultiPolygon([poly for i, poly in enumerate(multi_polys) if i in indices_to_keep])

    idx = gdf[gdf['state'] == 'Alaska'].index[0]
    gdf[gdf.index == idx] = gdf[gdf.index == idx].set_geometry([new_multi_polys])
    return gdf


def create_heatmap(df, var_name='z_score', center=0):
    """General function for creating sns heatmap for variables which measure the performance 
    of our forecasts with respect to other models

    Args:
        df (pd.DataFrame): The dataframe of where rows correspond to different states, 
        and the different columns are metrics for evaluating model performance. 
        Like z_score, non_param_z_score
        var_name (str, optional): Which column name to use. Defaults to 'z_score'.
        center (int, optional): Where is the heatmap to be centered. Defaults to 0.

    Returns:
        mpl.Figure, mpl.Axes: Figure and Axes of the heatmap plot
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    df_sorted = df.sort_values(var_name)
    df_sorted.dropna(axis=0, how='any', inplace=True)
    annot_mat = df_sorted.index.to_numpy().reshape(-1, 3) + " : " + \
    np.around(df_sorted[var_name].to_numpy().reshape(-1, 3), 2).astype(str)
    annot_mat = annot_mat.astype(str).tolist()
    sns.heatmap(df_sorted[var_name].to_numpy().reshape(-1, 3), 
                cmap='coolwarm', center=center, xticklabels=False,
                yticklabels=False, annot=annot_mat, fmt='', ax=ax)

    ax.set_title(
        f'Heatmap of {var_name.replace("_", " ")} for all US states, \n' + \
            f'sorted by {var_name.replace("_", " ")}, cmap centered at {center}')
    return fig, ax

def _label_geographies(ax, df_geo, var, adjust_text=False, remove_ticks=True):
    """Helper function for labelling all the geographies in the choropleth plot

    Args:
        ax (mpl.Axes): The axes in which annotation is to be done
        df_geo (gpd.GeoDataFrame): The gdf with the geometry data
        var (str): The variable which is to be annotated
        adjust_text (bool, optional): If true, location of annotation is adjusted. Defaults to False.
        remove_ticks (bool, optional): If true, removed the ticks of the axes. Defaults to True.
    """
    df_geo['centroid'] = df_geo['geometry'].centroid
    texts = []
    for _, row in df_geo.iterrows():
        if (not row['centroid'] is None) & (not pd.isna(row[var])):
            label = np.around(row[var], 2)
            texts.append(ax.text(row['centroid'].x,
                                 row['centroid'].y, label, fontsize=10))

    if adjust_text:
        aT.adjust_text(texts, force_points=0.3, force_text=0.8, expand_points=(1, 1), 
                       expand_text=(1, 1), arrowprops=dict(arrowstyle="-", color='black', lw=0.5))

    if remove_ticks:
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])


def create_single_choropleth(df, var='z_score', vcenter=0, vmin=-1, vmax=1, cmap='coolwarm', ax=None, gdf=None, 
                             adjust_text=False):
    """Function for creating single choropleth for all US states

    Args:
        df (pd.DataFrame): The dataframe with the data for which the choropleth is to be made
        var (str, optional): The variable to make the choropleth for. Defaults to 'z_score'.
        vcenter (int, optional): Center of cmap. Defaults to 0.
        vmin (int, optional): Min of cmap. Defaults to -1.
        vmax (int, optional): Max of cmap. Defaults to 1.
        cmap (str, optional): Which matplotlib cmap to use. Defaults to 'coolwarm'.
        ax (mpl.Axes, optional): If given, makes plot on this axes. Defaults to None.
        gdf (gpd.GeoDataFrame, optional): If given, uses this as the geodataframe. Defaults to None.
        adjust_text (bool, optional): If true, adjusts the text of the annotation. Defaults to False.

    Returns:
        [type]: [description]
    """
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    if gdf is None:
        gdf = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))
    df_geo = gdf.merge(df, left_on='state', right_index=True, how='left')
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 12))
    else:
        fig = None
    df_noncontigous = copy(df_geo[df_geo['state'].isin(['Alaska', 'Hawaii'])])
    df_geo = df_geo[np.logical_not(df_geo['state'].isin(['Alaska', 'Hawaii']))]
    df_geo.plot(column=var, cmap=cmap, linewidth=0.8,
                ax=ax, edgecolor='silver', norm=norm, 
                missing_kwds={"color": "darkgrey", "edgecolor": "red",
                              "hatch": "///"})
    subax_1 = add_inset_subplot_to_axes(ax, [0.2, 0.014, 0.12, 0.12])
    df_noncontigous[df_noncontigous['state'] == 'Hawaii'].plot(
        column=var, cmap=cmap, linewidth=0.8,
        ax=subax_1, edgecolor='silver', norm=norm
    )
    
    subax_2 = add_inset_subplot_to_axes(ax, [0, 0.014, 0.25, 0.25])
    df_noncontigous[df_noncontigous['state'] == 'Alaska'].plot(
        column=var, cmap=cmap, linewidth=0.8,
        ax=subax_2, edgecolor='silver', norm=norm
    )

    _label_geographies(ax, df_geo, var, adjust_text)
    _label_geographies(subax_1, copy(df_noncontigous[df_noncontigous['state'] == 'Hawaii']), var, adjust_text)
    _label_geographies(subax_2, copy(df_noncontigous[df_noncontigous['state'] == 'Alaska']), var, adjust_text)

    ax.set_title(f'Choropleth for {var.replace("_", " ")}')
    legend_elements = [Patch(facecolor='silver', edgecolor='r', hatch="///", label='Did Not Forecast')]
    ax.legend(handles=legend_elements)
    return fig

def plot_multiple_choropleths(df_wadhwani:pd.DataFrame, gdf:gpd.GeoDataFrame, vars_to_plot:dict):
    """Function for plotting multiple choropleths in 1 fig

    Args:
        df_wadhwani (pd.DataFrame): dataframe with different metrics analysing model performance for every location
        gdf (gpd.GeoDataFrame): The processed shapefile as a geopandas dataframe
        vars_to_plot (dict): Dict containing which params to plot choropleths for, and corresponding plotting params

    Returns:
        mpl.Figure, [mpl.Axes]: The matplotlib figure and axes object
    """
    fig, axs = plt.subplots(figsize=(14, 7*len(vars_to_plot)), 
                            nrows=len(vars_to_plot), ncols=1)
    for i, (var, plotting_params) in enumerate(vars_to_plot.items()):
        _ = create_single_choropleth(df_wadhwani, var=var, vcenter=plotting_params['vcenter'],
                                     vmin=plotting_params['vmin'], vmax=plotting_params['vmax'],
                                     cmap=plotting_params['cmap'], ax=axs.flat[i], gdf=gdf)
        fig.colorbar(cm.ScalarMappable(norm=colors.TwoSlopeNorm(
            vmin=plotting_params['vmin'], vcenter=plotting_params['vcenter'], 
            vmax=plotting_params['vmax']), cmap=plotting_params['cmap']), ax=axs.flat[i])
    
    return fig, axs

def create_scatter_plot_mape(df_wadhwani, annotate=True, abbv=False, abbv_dict=None, annot_z_score=False, 
                             stat_metric_to_use='z_score', log_scale=False):
    """Function for creating scatter plot of train error vs test error 

    Args:
        df_wadhwani (pd.DataFrame): The dataframe which contains the train and test error values
        annotate (bool, optional): If true, each of the points in the scatter 
        are annotated. Defaults to True.
        abbv (bool, optional): If true, the abbreviation of the state is used 
        for annotation. Defaults to False.
        abbv_dict (dict, optional): Dict mapping state name to state code name. 
        Necessary if abbv=True. Defaults to None.
        annot_z_score (bool, optional): If true, the Z score is also annotated. 
        Defaults to False.
        stat_metric_to_use (str, optional): Which statistical metric to use on the 
        color and radius scale. Defaults to 'z_score'
        log_scale (bool, optional): If true, the y_scale is lo. Defaults to False.

    Returns:
        mpl.Figure: Matplotlib figure with the scatter plot
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    df_zpos = df_wadhwani[df_wadhwani[stat_metric_to_use] > 0]
    df_zneg = df_wadhwani[df_wadhwani[stat_metric_to_use] < 0]
    ax.scatter(df_zpos['best_loss_train'], df_zpos['model_mape'],
            s=df_zpos[stat_metric_to_use]*100, c='red', marker='o', label='+ve Z score states')
    ax.scatter(df_zneg['best_loss_train'], df_zneg['model_mape'], s=-
            df_zneg[stat_metric_to_use]*100, c='blue', marker='o', label='-ve Z score states')
    if annotate:
        for i, (index, row) in enumerate(df_wadhwani.iterrows()):
            if abbv:
                state_name = abbv_dict[index]
            else:
                state_name = index
            if annot_z_score:
                annot_str = f'{state_name} ({round(row[stat_metric_to_use], 2)})'
            else:
                annot_str = f'{state_name}'
            ax.annotate(annot_str, (row['best_loss_train'], row['model_mape']))
    if log_scale:
        ax.set_yscale('log')
    ax.set_xlabel('MAPE on training data (calculated daily)')
    ax.set_ylabel(f'MAPE on unseen data (calculated weekly) ({"log" if log_scale else "linear"} scale)')
    ax.axvline(1, ls=':', c='red', label='train error threshold (1\%)')
    ax.axhline(5, ls=':', c='blue', label='test error threshold (5\%)')
    ax.legend()
    ax.set_title(f'Scatter plot of train vs test error, point radii proportional to {stat_metric_to_use.replace("_", " ")}')
    return fig


def create_scatter_plot_zscores(df_wadhwani, annotate=False, us_states_abbv_dict=None):
    """Create scatter plot between Z score and Non Param Z Score

    Args:
        df_wadhwani (pd.DataFrame): Dataframe which has Z and non param Z scores for all states
        annotate (bool, optional): If true, annotates scatters with all state code. Defaults to False.
        us_states_abbv_dict (dict, optional): if annotate=true, uses this state_code dict to annotate. 
        Necessary if annotate=True. Defaults to None.

    Returns:
        mpl.Figure, mpl.Axes: The figure and axes object
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(df_wadhwani['z_score'], df_wadhwani['non_param_z_score'])
    ax.plot(df_wadhwani['z_score'], df_wadhwani['z_score'], '-r', label='y=x line')
    ax.set_xlabel('Z Score (Mean, Std)')
    ax.set_ylabel('Non Param Z Score (Median, MAD)')
    ax.set_title('Non Param Z Score vs Z Score ')
    ax.grid()
    if annotate:
        for  index, row in df_wadhwani.iterrows():
            annot_str = us_states_abbv_dict[index]
            ax.annotate(annot_str, (row['z_score'], row['non_param_z_score']))
    ax.legend()

    return fig, ax

def plot_ecdf_single_state(df_mape, state, ax, model='Wadhwani_AI'):
    """Function for ECDF plot for a single region 

    Args:
        df_mape (pd.DataFrame): The dataframe with MAPE values for every model
        state (str): The state for which the plotting has to done
        ax (mpl.Axes): The axes object to make the plot in
        model (str, optional): Which model to highlight via a vertical line. Defaults to 'Wadhwani_AI'.
    """
    sns.ecdfplot(data=df_mape[state], ax=ax)
    ax.axvline(df_mape.loc[model, state], ls=':',
               c='red', label='Wadhwani AI Submission')
    ax.set_title(state)
    ax.set_xlabel('MAPE')
    ax.legend()


def plot_ecdf_all_states(df_mape):
    """Function for ECDF plots for all states

    Args:
        df_mape (pd.DataFrame): The dataframe with MAPE values for every model, region

    Returns:
        mpl.Figure, [mpl.Axes] : The figure and axes objects
    """
    fig, axs = plt.subplots(figsize=(21, 6*15), nrows=15, ncols=3)
    columns = df_mape.loc[:, np.logical_not(
        df_mape.loc['Wadhwani_AI', :].isna())].columns
    for i, state in enumerate(columns):
        ax = axs.flat[i]
        plot_ecdf_single_state(df_mape, state, ax, model='Wadhwani_AI')
    fig.suptitle('Emperical Cumulative Distribution Function Plots for all states')
    fig.subplots_adjust(top=0.97)

    return fig, axs

def plot_qq_single_state(df_mape, state, ax, fit=False, df_wadhwani=None):
    """Function for QQ plot for a single region 

    Args:
        df_mape (pd.DataFrame): The dataframe with MAPE values for every model, region
        state (str): The state for which the plotting has to done
        ax (mpl.Axes): The axes object to make the plot in
        fit (bool, optional): If true, theoretical quantiles are fit to data. Defaults to False.
        df_wadhwani (pd.DataFrame, optional): If fit=true, the mean and std values 
        to use for the theoretical quantiles. Defaults to None.
    """

    if fit:
        sm.qqplot(df_mape[state], dist=stats.norm, fit=True, line='45', ax=ax)
    else:
        sm.qqplot(df_mape[state], dist=stats.norm, loc=df_wadhwani.loc[state, 'mean_mape'], 
                scale=df_wadhwani.loc[state, 'std_mape'], line='45', ax=ax)
    ax.set_title(state)
    

def plot_qq_all_states(df_mape, fit=True, df_wadhwani=None):
    """Function for QQ plot for a single region 

    Args:
        df_mape (pd.DataFrame): The dataframe with MAPE values for every model, region
        fit (bool, optional): If true, theoretical quantiles are fit to data. Defaults to False.
        df_wadhwani (pd.DataFrame, optional): If fit=true, the mean and std values 
        to use for the theoretical quantiles. Defaults to None.
    
    Returns:
        mpl.Figure, [mpl.Axes] : The figure and axes objects
    """
    fig, axs = plt.subplots(figsize=(21, 6*15), nrows=15, ncols=3)
    columns = df_mape.loc[:, np.logical_not(
        df_mape.loc['Wadhwani_AI', :].isna())].columns
    for i, state in enumerate(columns):
        ax = axs.flat[i]
        plot_qq_single_state(df_mape, state, ax, fit=fit,
                             df_wadhwani=df_wadhwani)
    fig.suptitle('Q-Q plots for all states')
    fig.subplots_adjust(top=0.97)

    return fig, axs


def mape_heatmap_single_state(df_mape, state, cmap='Reds'):
    """Function for creating rectangular heatmap of MAPE values of models for given state

    Args:
        df_mape (pd.DataFrame): dataframe of mape values for all models, region
        state (str): Which state to make heatmap for
        cmap (str, optional): What colormap to use. Defaults to 'Reds'.

    Returns:
        mpl.Figure, mpl.Axes: The Figure and Axes object
    """
    fig, ax = plt.subplots(figsize=(18, 2))
    sns.heatmap(df_mape[state].to_numpy().reshape(1, -1), cmap=cmap,
                ax=ax, xticklabels=df_mape[state].index, annot=True)
    ax.set_title(state)

    return fig, ax


def qtile_barchart(df_comb, quant, color='C0', latex=False):
    df_temp = df_comb[df_comb['quantile'] == quant]

    df_mape = df_temp.groupby(['model', 'location',
                               'location_name']).mean().reset_index()

    df_mape = df_mape.pivot(index='model', columns='location_name',
                            values='perc_loss_mape')

    df_plot = pd.DataFrame([df_mape.median(axis=1)]).T
    df_plot.columns = ['median']
    df_plot['mad'] = median_abs_deviation(df_mape, axis=1, nan_policy='omit')
    df_plot = df_plot.sort_values('median')

    fig, ax = plt.subplots(figsize=(12, 12))

    y_pos = np.arange(len(df_plot))
    barlist = ax.barh(y_pos, df_plot['median'],
                      xerr=df_plot['mad'], align='center', color=color)
    ax.set_yticks(y_pos)
    if latex:
        ax.set_yticklabels([x.replace('_', '\_') for x in df_plot.index])
    ax.invert_yaxis()  # labels read top-to-bottom
    wai_idx = np.where(df_plot.index == 'Wadhwani_AI')[0][0]
    barlist[wai_idx].set_color('purple')
    ax.set_xlabel('Median of Quantile MAPE Loss aggregated across states')
    ax.set_title(
        f'Median Quantile MAPE loss for {int(quant*100)}th percentile for all models')

    return fig, ax
