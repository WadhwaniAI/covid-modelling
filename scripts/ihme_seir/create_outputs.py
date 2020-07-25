import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import json
import ast

sys.path.append('../../')

from main.ihme_seir.synthetic_data_generator import create_output_folder


compartments = ['hospitalised', 'deceased', 'recovered', 'total_infected']
c1 = ['orange', 'lightcoral', 'lime', 'lightblue']
c2 = ['darkviolet', 'indigo', 'darkgreen', 'blue']
loss_functions = ['mape', 'rmse', 'rmsle']
ptwise_loss_functions = ['ape', 'error', 'se', 'sle']
data_split = ['train', 'val']
titles = ['Using ground truth train data', 'Using IHME forecast as train data', 'Using SEIR forecast as train data',
          'Baseline']


def colorFader(c1, c2, mix=0.0):
    c1=np.array(matplotlib.colors.to_rgb(c1))
    c2=np.array(matplotlib.colors.to_rgb(c2))
    return matplotlib.colors.to_hex((1-mix)*c1 + mix*c2)


def get_ihme_loss_dict(path, file, num):
    loss_dict = dict()
    for i in range(num):
        loss_dict[i] = pd.read_csv(f'{path}/{str(i)}/{file}',
                                   index_col=['compartment', 'split', 'loss_function'])
    return loss_dict


def get_seir_loss_dict(path, file, num):
    loss_dict = dict()
    for i in range(num):
        loss_dict[i] = pd.read_csv(f'{path}/{str(i)}/{file}', index_col=0)
    return loss_dict


def get_seir_pointwise_loss_dict(path, file, num):
    loss_dict = dict()
    for i in range(num):
        loss_dict[i] = pd.read_csv(f'{path}/{str(i)}/{file}', index_col=['compartment', 'loss_function'])
    return loss_dict


def get_best_params(path, file, num, dates, var_param_ranges_file=None):
    param_ranges_df = pd.DataFrame
    if var_param_ranges_file is not None:
        with open(f'{path}/0/{var_param_ranges_file}', 'r') as infile:
            var_param_ranges_dict = json.load(infile)
            if isinstance(var_param_ranges_dict, str):
                var_param_ranges_dict = ast.literal_eval(var_param_ranges_dict)
        param_ranges_df = pd.DataFrame.from_dict(var_param_ranges_dict, orient="columns")
        param_ranges_df.index = ["lower", "upper"]
        param_ranges_df.insert(0, column="Training start date", value=None)
    param_dict = dict()
    for i in range(num):
        with open(f'{path}/{str(i)}/{file}', 'r') as infile:
            param_dict[i] = json.load(infile)
    params_df = pd.DataFrame.from_dict(param_dict, orient="index")
    params_df.insert(0, column='Training start date', value=dates)
    params_df = pd.concat([param_ranges_df, params_df], axis=0)
    return params_df


def save_best_params_df(path, output_folder, num, dates):
    files = [f'{name_prefix}_c1_best_params.json' for name_prefix in ['seirt', 'sird']]
    ranges_files = [f'{name_prefix}_variable_param_ranges.json' for name_prefix in ['seirt', 'sird']]
    for name_prefix in ['seirt', 'sird']:
        for exp in [1, 2, 3]:
            files.append(f'{name_prefix}_c2_best_params_exp_{str(exp)}.json')
            ranges_files.append(f'{name_prefix}_variable_param_ranges.json')
    params_df_dict = dict()
    for i in range(len(files)):
        params_df_dict[files[i]] = get_best_params(path, files[i], num, dates, var_param_ranges_file=ranges_files[i])
    params_df = pd.concat(params_df_dict.values(), keys=params_df_dict.keys(), axis=0)
    params_df.to_csv(f'{output_folder}/params.csv')


def save_ihme_model_params(path, output_folder, num, dates):
    file = 'ihme_i1_model_params.json'
    param_dict = dict()
    param_ranges = None
    for i in range(num):
        with open(f'{path}/{str(i)}/{file}', 'r') as infile:
            param_dict[i] = json.load(infile)
            if isinstance(param_dict[i], str):
                params_str = param_dict[i][param_dict[i].find("'priors'"):]
                params_str = "".join(['{', params_str[:params_str.find('}')+1], '}'])
                param_dict[i] = ast.literal_eval(params_str)['priors']
                if i == 0:
                    param_ranges = param_dict[i]['fe_bounds']
                param_dict[i] = param_dict[i]['fe_init']
    params_df = pd.DataFrame.from_dict(param_dict, orient="index")
    params_df.columns = ['alpha', 'beta', 'p']
    params_df.insert(0, column='Training start date', value=dates)

    param_ranges_df = pd.DataFrame(param_ranges).T
    param_ranges_df.columns = ['alpha', 'beta', 'p']
    param_ranges_df.index = ['lower', 'upper']
    param_ranges_df.insert(0, column="Training start date", value=None)
    params_df = pd.concat([param_ranges_df, params_df], axis=0)
    params_df.to_csv(f'{output_folder}/ihme_params.csv')


def get_ihme_loss(loss_dict, compartment, split, loss_fn):
    losses = []
    for i in loss_dict:
        losses.append(loss_dict[i].loc[(compartment, split, loss_fn)]['loss'])
    return losses


def get_ihme_pointwise_loss(loss_dict, compartment, split, loss_fn):
    losses = []
    for i in loss_dict:
        losses.append(loss_dict[i].loc[(compartment, split, loss_fn)])
    return losses


def get_seir_loss(loss_dict, compartment, split):
    losses = []
    for i in loss_dict:
        losses.append(loss_dict[i].loc[split][compartment])
    return losses


def get_seir_pointwise_loss(loss_dict, compartment, loss_fn):
    losses = []
    for i in loss_dict:
        losses.append(loss_dict[i].loc[(compartment, loss_fn)])
    return losses


def get_seir_loss_exp(loss_dict, compartment, split, exp):
    losses = []
    for i in loss_dict:
        df = loss_dict[i].loc[split]
        losses.append(df[df['exp'] == exp][compartment])
    return losses


def get_error_diff(pointwise_loss_dict1, pointwise_loss_dict2, compartment, loss_fn, shift):
    total_diff = np.zeros(shift)
    total = np.array([len(pointwise_loss_dict1)]*shift)
    for i in range(len(pointwise_loss_dict1)):
        exp_losses = pointwise_loss_dict1[i].loc[(compartment, loss_fn)].values
        baseline_losses = pointwise_loss_dict2[i].loc[(compartment, loss_fn)]
        baseline_losses = baseline_losses.iloc[-len(exp_losses):].values
        diff = np.subtract(baseline_losses[:shift], exp_losses[:shift])
        diff = np.where(diff >= 0, 1, 0)  # if diff >= 0, baseline loss is higher and we add 1 for the experiment
        total_diff += diff
    return total_diff/total


def save_error_diff(path, output_folder, num, shift):
    c3_pointwise_loss_dict = get_seir_pointwise_loss_dict(path, 'seirt_c3_pointwise_val_loss.csv', num)
    shift_range = [f'day_{i}' for i in range(1, shift + 1)]
    col_index = pd.MultiIndex.from_product([compartments, shift_range])
    comparison_df = pd.DataFrame(columns=col_index, index=[1, 2, 3])
    for exp in range(3):
        c2_pointwise_loss_dict = get_seir_pointwise_loss_dict(path, f'seirt_c2_pointwise_val_loss_exp_{exp + 1}.csv',
                                                              num)
        for compartment in compartments:
            comparison_df.loc[exp + 1, :][compartment] = \
                get_error_diff(c2_pointwise_loss_dict, c3_pointwise_loss_dict, compartment, 'ape', shift)

    comparison_df.to_csv(f'{output_folder}/seirt_c2_experiment_baseline_comparison.csv')


def all_outputs(root_folder, region, num, shift, start_date):
    path = f'../../outputs/ihme_seir/synth/{root_folder}'
    output_path = create_output_folder(f'../../outputs/consolidated/{root_folder}')

    start_date = pd.to_datetime(start_date, dayfirst=False)
    dates = pd.date_range(start=start_date, periods=num, freq=f'{shift}D').tolist()
    dates = [date.strftime("%m-%d-%Y") for date in dates]

    # Day by day comparison of whether experiments do better than the baseline
    save_error_diff(path, output_path, num, shift)

    # Save IHME model params
    save_ihme_model_params(path, output_path, num, dates)

    # Save model params
    save_best_params_df(path, output_path, num, dates)

    # PLOT: IHME I1 plot (train and val)
    print("IHME I1 plot of train and val MAPE on s1 and s2 respectively")
    i1_loss_dict = get_ihme_loss_dict(path, 'ihme_i1_loss.csv', num)
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(15, 10))
    fig.suptitle(f'{region}: IHME MAPE on s2')
    for i, compartment in enumerate(compartments):
        train_losses = get_ihme_loss(i1_loss_dict, compartment, 'train', 'mape')
        val_losses = get_ihme_loss(i1_loss_dict, compartment, 'val', 'mape')
        ax[i//2][i % 2].plot(dates, train_losses, 'o-', label='IHME train MAPE on s1', color='blue')
        ax[i//2][i % 2].plot(dates, val_losses, 'o-', label='IHME test MAPE on s2', color='red')
        ax[i//2][i % 2].title.set_text(compartment)
        ax[i//2][i % 2].set_xlabel(f'Training start date', fontsize=12)
        ax[i//2][i % 2].set_ylabel('MAPE', fontsize=12)
        ax[i//2][i % 2].grid()
        ax[i//2][i % 2].tick_params(labelrotation=45)
    plt.legend()
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    plt.savefig(f'{output_path}/i1_train_val.png')
    plt.close()

    # PLOT: IHME I1, SEIHRD C1 AND SIRD C1 (val - s2)
    print("IHME I1, SEIR_Testing C1 and SIRD C1 val losses on s2")
    seirt_c1_loss = get_seir_loss_dict(path, 'seirt_c1_loss.csv', num)
    sird_c1_loss = get_seir_loss_dict(path, 'sird_c1_loss.csv', num)
    fig, ax = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(20, 15))
    fig.suptitle(f'{region}: MAPE on s2')
    for i, compartment in enumerate(compartments):
        ihme_losses = get_ihme_loss(i1_loss_dict, compartment, 'val', 'mape')
        seirt_losses = get_seir_loss(seirt_c1_loss, compartment, 'val')
        sird_losses = get_seir_loss(sird_c1_loss, compartment, 'val')
        ax[i].plot(dates, ihme_losses, 'o-', label='IHME test MAPE on s2', color='red')
        ax[i].plot(dates, seirt_losses, 'o-', label='SEIRT test MAPE on s2', color='blue')
        ax[i].plot(dates, sird_losses, 'o-', label='SIRD test MAPE on s2', color='green')
        ax[i].title.set_text(compartment)
        ax[i].set_xlabel(f'Training start date', fontsize=12)
        ax[i].set_ylabel('MAPE', fontsize=12)
        ax[i].grid()
        ax[i].tick_params(labelrotation=45)
        ax[i].legend()
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    plt.savefig(f'{output_path}/ihme_i1_seirt_c1_sird_c1_val.png')
    plt.close()

    # PLOT: IHME I1, SEIHRD C1 AND SIRD C1 (pointwise)
    print("IHME I1, SEIR_Testing C1 and SIRD C1 pointwise val losses on s2")
    i1_pointwise_loss_dict = get_ihme_loss_dict(path, 'ihme_i1_pointwise_test_loss.csv', num)
    seirt_c1_pointwise_loss_dict = get_seir_pointwise_loss_dict(path, 'seirt_c1_pointwise_val_loss.csv', num)
    sird_c1_pointwise_loss_dict = get_seir_pointwise_loss_dict(path, 'sird_c1_pointwise_val_loss.csv', num)
    model_type = ['IHME', 'SEIR_Testing', 'SIRD']
    for i, compartment in enumerate(compartments):
        fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(20, 15))
        fig.suptitle(f'{region} - {compartment} - pointwise test APE on s2')
        losses = dict()
        # Change this from 'val_no_xform' to 'val' if correction is made in main.ihme.fitting
        losses[0] = get_ihme_pointwise_loss(i1_pointwise_loss_dict, compartment, 'val_no_xform', 'ape')
        losses[1] = get_seir_pointwise_loss(seirt_c1_pointwise_loss_dict, compartment, 'ape')
        losses[2] = get_seir_pointwise_loss(sird_c1_pointwise_loss_dict, compartment, 'ape')
        for k in range(3):
            ax[k].title.set_text(model_type[k])
            for j, l in enumerate(losses[k]):
                ax[k].plot(l, '-o', color=colorFader(c1[i], c2[i], j / num),
                        label=f'Test APE for {l.index[0]} to {l.index[-1]}')
            ax[k].set_xlabel('Date', fontsize=12)
            ax[k].set_ylabel('APE', fontsize=12)
            ax[k].tick_params(labelrotation=45)
            ax[k].grid()
        plt.legend()
        fig.tight_layout()
        fig.subplots_adjust(top=0.94)
        plt.savefig(f'{output_path}/ihme_i1_seirt_c1_sird_c1_pointwise_val_{compartment}.png')
        plt.close()

    # PLOT: SEIHRD C1 AND SIRD C1 (val - s3)
    print("SEIR_Testing C1 and SIRD C1 val losses on s3")
    seirt_c3_loss = get_seir_loss_dict(path, 'seirt_baseline_loss.csv', num)
    sird_c3_loss = get_seir_loss_dict(path, 'sird_baseline_loss.csv', num)
    fig, ax = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(20, 15))
    fig.suptitle(f'{region}: MAPE on s3 using baseline model')
    for i, compartment in enumerate(compartments):
        seirt_losses = get_seir_loss(seirt_c3_loss, compartment, 'val')
        sird_losses = get_seir_loss(sird_c3_loss, compartment, 'val')
        ax[i].plot(dates, seirt_losses, 'o-', label='SEIRT test MAPE on s3', color='blue')
        ax[i].plot(dates, sird_losses, 'o-', label='SIRD test MAPE on s3', color='green')
        ax[i].title.set_text(compartment)
        ax[i].set_xlabel(f'Training start date', fontsize=12)
        ax[i].set_ylabel('MAPE', fontsize=12)
        ax[i].grid()
        ax[i].tick_params(labelrotation=45)
        ax[i].legend()
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    plt.savefig(f'{output_path}/seirt_c3_sird_c3_val.png')
    plt.close()

    # PLOT: Synthetic data experiments with SEIR_Testing
    print("SEIR_Testing C2 on s3")
    seirt_c2_loss = get_seir_loss_dict(path, 'seirt_experiments_loss.csv', num)
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(15, 10))
    fig.suptitle(f'{region}: MAPE on s3 using synthetic data')
    for i, compartment in enumerate(compartments):
        colors = ['green', 'red', 'blue']
        for j in range(3):
            seirt_exp_losses = get_seir_loss_exp(seirt_c2_loss, compartment, 'val', j+1)
            ax[i//2][i % 2].plot(dates, seirt_exp_losses, 'o-', label=f'SEIRT test MAPE on s3 {titles[j]}', color=colors[j])
        seirt_baseline_losses = get_seir_loss(seirt_c3_loss, compartment, 'val')
        ax[i//2][i % 2].plot(dates, seirt_baseline_losses, 'o-', label='SEIRT baseline test MAPE on s3', color='black')
        ax[i//2][i % 2].title.set_text(compartment)
        ax[i//2][i % 2].set_xlabel(f'Training start date', fontsize=12)
        ax[i//2][i % 2].set_ylabel('MAPE', fontsize=12)
        ax[i//2][i % 2].grid()
        ax[i//2][i % 2].tick_params(labelrotation=45)
    plt.legend()
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    plt.savefig(f'{output_path}/seirt_experiments_val.png')
    plt.close()

    # PLOT: Synthetic data experiments with SEIR_Testing (4X1)
    print("SEIR_Testing C2 on s3")
    seirt_c2_loss = get_seir_loss_dict(path, 'seirt_experiments_loss.csv', num)
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(15, 10))
    fig.suptitle(f'{region}: MAPE on s3 using synthetic data')
    for i, compartment in enumerate(compartments):
        colors = ['green', 'red', 'blue']
        for j in range(3):
            seirt_exp_losses = get_seir_loss_exp(seirt_c2_loss, compartment, 'val', j + 1)
            ax[i].plot(dates, seirt_exp_losses, 'o-', label=f'SEIRT test MAPE on s3 {titles[j]}',
                                   color=colors[j])
        seirt_baseline_losses = get_seir_loss(seirt_c3_loss, compartment, 'val')
        ax[i].plot(dates, seirt_baseline_losses, 'o-', label='SEIRT baseline test MAPE on s3',
                               color='black')
        ax[i].title.set_text(compartment)
        ax[i].set_xlabel(f'Training start date', fontsize=12)
        ax[i].set_ylabel('MAPE', fontsize=12)
        ax[i].grid()
        ax[i].tick_params(labelrotation=45)
    plt.legend()
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    plt.savefig(f'{output_path}/seirt_experiments_v2_val.png')
    plt.close()

    # PLOT: Synthetic data experiments with SIRD
    print("SIRD C2 on s3")
    sird_c2_loss = get_seir_loss_dict(path, 'sird_experiments_loss.csv', num)
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(15, 10))
    fig.suptitle(f'{region}: MAPE on s3 using synthetic data')
    for i, compartment in enumerate(compartments):
        colors = ['green', 'red', 'blue']
        for j in range(3):
            sird_exp_losses = get_seir_loss_exp(sird_c2_loss, compartment, 'val', j + 1)
            ax[i//2][i % 2].plot(dates, sird_exp_losses, 'o-', label=f'SIRD test MAPE on s3 {titles[j]}', color=colors[j])
        sird_baseline_losses = get_seir_loss(sird_c3_loss, compartment, 'val')
        ax[i//2][i % 2].plot(dates, sird_baseline_losses, 'o-', label='SIRD baseline test MAPE on s3', color='black')
        ax[i//2][i % 2].title.set_text(compartment)
        ax[i//2][i % 2].set_xlabel(f'Training start date', fontsize=12)
        ax[i//2][i % 2].set_ylabel('MAPE', fontsize=12)
        ax[i//2][i % 2].grid()
        ax[i//2][i % 2].tick_params(labelrotation=45)
    plt.legend()
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    plt.savefig(f'{output_path}/sird_experiments_val.png')
    plt.close()

    # PLOT: Synthetic data experiments with SIRD (4X1)
    print("SIRD C2 on s3")
    sird_c2_loss = get_seir_loss_dict(path, 'sird_experiments_loss.csv', num)
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(15, 10))
    fig.suptitle(f'{region}: MAPE on s3 using synthetic data')
    for i, compartment in enumerate(compartments):
        colors = ['green', 'red', 'blue']
        for j in range(3):
            sird_exp_losses = get_seir_loss_exp(sird_c2_loss, compartment, 'val', j + 1)
            ax[i].plot(dates, sird_exp_losses, 'o-', label=f'SIRD test MAPE on s3 {titles[j]}',
                       color=colors[j])
        sird_baseline_losses = get_seir_loss(seirt_c3_loss, compartment, 'val')
        ax[i].plot(dates, sird_baseline_losses, 'o-', label='SIRD baseline test MAPE on s3',
                   color='black')
        ax[i].title.set_text(compartment)
        ax[i].set_xlabel(f'Training start date', fontsize=12)
        ax[i].set_ylabel('MAPE', fontsize=12)
        ax[i].grid()
        ax[i].tick_params(labelrotation=45)
    plt.legend()
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    plt.savefig(f'{output_path}/sird_experiments_v2_val.png')
    plt.close()

    # PLOT: IHME I1 pointwise - val
    i1_pointwise_loss_dict = get_ihme_loss_dict(path, 'ihme_i1_pointwise_test_loss.csv', num)
    print("IHME pointwise test APE on s2")
    for i, compartment in enumerate(compartments):
        fig, ax = plt.subplots(1, figsize=(20, 10))
        fig.suptitle(f'{region} - {compartment} - IHME pointwise test APE on s2')
        val_losses = get_ihme_pointwise_loss(i1_pointwise_loss_dict, compartment, 'val_no_xform', 'ape')  # Check
        for j, l in enumerate(val_losses):
            ax.plot(l, '-o', color=colorFader(c1[i], c2[i], j/num),
                    label=f'Test APE for {l.index[0]} to {l.index[-1]}')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('APE', fontsize=12)
            ax.tick_params(labelrotation=45)
        plt.legend()
        plt.grid()
        fig.tight_layout()
        fig.subplots_adjust(top=0.94)
        plt.savefig(f'{output_path}/ihme_i1_test_pointwise_{compartment}.png')
        plt.close()

    # PLOT: IHME I1 pointwise - train
    i1_pointwise_loss_dict = get_ihme_loss_dict(path, 'ihme_i1_pointwise_train_loss.csv', num)
    print("IHME pointwise train APE on s1")
    for i, compartment in enumerate(compartments):
        fig, ax = plt.subplots(1, figsize=(20, 10))
        fig.suptitle(f'{region} - {compartment} - IHME pointwise train APE on s1')
        train_losses = get_ihme_pointwise_loss(i1_pointwise_loss_dict, compartment, 'train_no_xform', 'ape')  # Check
        for j, l in enumerate(train_losses):
            ax.plot(l, '-o', color=colorFader(c1[i], c2[i], j/num),
                    label=f'Train APE for {l.index[0]} to {l.index[-1]}')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('APE', fontsize=12)
            ax.tick_params(labelrotation=45)
        plt.legend()
        plt.grid()
        fig.tight_layout()
        fig.subplots_adjust(top=0.94)
        plt.savefig(f'{output_path}/ihme_i1_train_pointwise_{compartment}.png')
        plt.close()

    # PLOT: SEIRT C1 pointwise - val
    seirt_c1_pointwise_loss_dict = get_seir_pointwise_loss_dict(path, 'seirt_c1_pointwise_val_loss.csv', num)
    print("SEIR_Testing pointwise test APE on s2")
    for i, compartment in enumerate(compartments):
        fig, ax = plt.subplots(1, figsize=(20, 10))
        fig.suptitle(f'{region} - {compartment} - SEIR_Testing pointwise test APE on s2')
        val_losses = get_seir_pointwise_loss(seirt_c1_pointwise_loss_dict, compartment, 'ape')
        for j, l in enumerate(val_losses):
            ax.plot(l, '-o', color=colorFader(c1[i], c2[i], j / num),
                    label=f'Test APE for {l.index[0]} to {l.index[-1]}')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('APE', fontsize=12)
            ax.tick_params(labelrotation=45)
        plt.grid()
        plt.legend()
        fig.tight_layout()
        fig.subplots_adjust(top=0.94)
        plt.savefig(f'{output_path}/seirt_c1_test_pointwise_{compartment}.png')
        plt.close()

    # PLOT: SEIRT C1 pointwise - train
    seirt_c1_pointwise_loss_dict = get_seir_pointwise_loss_dict(path, 'seirt_c1_pointwise_train_loss.csv', num)
    print("SEIR_Testing pointwise train APE on s1")
    for i, compartment in enumerate(compartments):
        fig, ax = plt.subplots(1, figsize=(20, 10))
        fig.suptitle(f'{region} - {compartment} - SEIR_Testing pointwise train APE on s1')
        train_losses = get_seir_pointwise_loss(seirt_c1_pointwise_loss_dict, compartment, 'ape')
        for j, l in enumerate(train_losses):
            ax.plot(l, '-o', color=colorFader(c1[i], c2[i], j / num),
                    label=f'Train APE for {l.index[0]} to {l.index[-1]}')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('APE', fontsize=12)
            ax.tick_params(labelrotation=45)
        plt.grid()
        plt.legend()
        fig.tight_layout()
        fig.subplots_adjust(top=0.94)
        plt.savefig(f'{output_path}/seirt_c1_train_pointwise_{compartment}.png')
        plt.close()

    # PLOT: SIRD C1 pointwise -  val
    sird_c1_pointwise_loss_dict = get_seir_pointwise_loss_dict(path, 'sird_c1_pointwise_val_loss.csv', num)
    print("SIRD pointwise test APE on s2")
    for i, compartment in enumerate(compartments):
        fig, ax = plt.subplots(1, figsize=(20, 10))
        fig.suptitle(f'{region} - {compartment} - SIRD pointwise test APE on s2')
        val_losses = get_seir_pointwise_loss(sird_c1_pointwise_loss_dict, compartment, 'ape')
        for j, l in enumerate(val_losses):
            ax.plot(l, '-o', color=colorFader(c1[i], c2[i], j / num),
                    label=f'Test APE for {l.index[0]} to {l.index[-1]}')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('APE', fontsize=12)
            ax.tick_params(labelrotation=45)
        plt.grid()
        plt.legend()
        fig.tight_layout()
        fig.subplots_adjust(top=0.94)
        plt.savefig(f'{output_path}/sird_c1_test_pointwise_{compartment}.png')
        plt.close()

    # PLOT: SIRD C1 pointwise -  train
    sird_c1_pointwise_loss_dict = get_seir_pointwise_loss_dict(path, 'sird_c1_pointwise_train_loss.csv', num)
    print("SIRD pointwise train APE on s1")
    for i, compartment in enumerate(compartments):
        fig, ax = plt.subplots(1, figsize=(20, 10))
        fig.suptitle(f'{region} - {compartment} - SIRD pointwise train APE on s1')
        train_losses = get_seir_pointwise_loss(sird_c1_pointwise_loss_dict, compartment, 'ape')
        for j, l in enumerate(train_losses):
            ax.plot(l, '-o', color=colorFader(c1[i], c2[i], j / num),
                    label=f'Train APE for {l.index[0]} to {l.index[-1]}')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('APE', fontsize=12)
            ax.grid()
            ax.tick_params(labelrotation=45)
        plt.legend()
        fig.tight_layout()
        fig.subplots_adjust(top=0.94)
        plt.savefig(f'{output_path}/sird_c1_train_pointwise_{compartment}.png')
        plt.close()

    # EXPERIMENT POINTWISE LOSSES

    print("All experiments pointwise")  # Number of plots = 2 (model variants) * 2 (splits) * 4 (compartments) = 16
    for name_prefix in ['seirt', 'sird']:
        for split in ['train', 'val']:
            for i, compartment in enumerate(compartments):
                if split == 'train':
                    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(20, 15))
                else:
                    fig, ax = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(20, 15))
                fig.suptitle(f'{region}: {name_prefix} pointwise {split} APE on s3', size=16)
                losses = None
                for exp in range(3):
                    c2_pointwise_loss_dict = get_seir_pointwise_loss_dict(
                        path, f'{name_prefix}_c2_pointwise_{split}_loss_exp_{exp+1}.csv', num)
                    losses = get_seir_pointwise_loss(c2_pointwise_loss_dict, compartment, 'ape')
                    for j, l in enumerate(losses):
                        ax[exp].plot(l, '-o', color=colorFader(c1[i], c2[i], j / num),
                                     label=f'{split} APE for {l.index[0]} to {l.index[-1]}')
                    ax[exp].title.set_text(f'{compartment}: {titles[exp]}')
                    ax[exp].set_xlabel('Date', fontsize=12)
                    ax[exp].set_ylabel('APE', fontsize=12)
                    ax[exp].tick_params(labelrotation=45)
                    ax[exp].grid()

                if split == 'val':
                    c3_pointwise_loss_dict = get_seir_pointwise_loss_dict(
                        path, f'{name_prefix}_c3_pointwise_{split}_loss.csv', num)
                    baseline_losses = get_seir_pointwise_loss(c3_pointwise_loss_dict, compartment, 'ape')
                    baseline_losses = [loss.iloc[-len(losses[0]):] for loss in baseline_losses]
                    for j, l in enumerate(baseline_losses):
                        ax[3].plot(l, '-o', color=colorFader(c1[i], c2[i], j / num),
                                   label=f'{split} APE for {l.index[0]} to {l.index[-1]}')
                    ax[3].title.set_text(f'{compartment}: {titles[3]}')
                    ax[3].set_xlabel('Date', fontsize=12)
                    ax[3].set_ylabel('APE', fontsize=12)
                    ax[3].tick_params(labelrotation=45)
                    ax[3].grid()
                fig.tight_layout()
                fig.subplots_adjust(top=0.94)
                plt.legend()
                plt.savefig(f'{output_path}/{name_prefix}_c2_{split}_pointwise_{compartment}_experiments_val.png')
                plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", help="folder to results relative to ihme_seir/synth/", required=True)
    parser.add_argument("-r", "--region", help="region name", required=True)
    parser.add_argument("-n", "--num", help="number of experiments", required=True)
    parser.add_argument("-s", "--shift", help="number of days to shift forward", required=True)
    parser.add_argument("-d", "--start_date", help="first date used", required=True)
    args = parser.parse_args()
    all_outputs(args.folder, args.region, int(args.num), int(args.shift), args.start_date)
