import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../../')

from main.ihme_seir.synthetic_data_generator import create_output_folder


compartments = ['hospitalised', 'deceased', 'recovered', 'total_infected']
loss_functions = ['mape', 'rmse', 'rmsle']
ptwise_loss_functions = ['ape', 'error', 'se', 'sle']
data_split = ['train', 'val']


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


def all_plots(root_folder, num, shift, start_date):
    path = f'../../outputs/ihme_seir/synth/{root_folder}'
    output_path = create_output_folder(f'../../outputs/consolidated/{root_folder}')

    start_date = pd.to_datetime(start_date, dayfirst=False)
    dates = pd.date_range(start=start_date, periods=num, freq='5D').tolist()

    dates = [date.strftime("%m-%d-%Y") for date in dates]
    start_date = start_date.strftime("%m-%d-%Y")

    # IHME I1 plot (train and val)
    print("IHME I1 plot of train and val MAPE on s1 ans s2 respectively")
    i1_loss_dict = get_ihme_loss_dict(path, 'ihme_i1_loss.csv', num)
    fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(20, 3))
    plt.title('MAPE on s2')
    for i, compartment in enumerate(compartments):
        train_losses = get_ihme_loss(i1_loss_dict, compartment, 'train', 'mape')
        val_losses = get_ihme_loss(i1_loss_dict, compartment, 'val', 'mape')
        ax[i].plot(dates, train_losses, 'o-', label='IHME train MAPE on s1', color='blue')
        ax[i].plot(dates, val_losses, 'o-', label='IHME test MAPE on s2', color='red')
        ax[i].title.set_text(compartment)
        ax[i].set_xlabel(f'Training start date (from {start_date})', fontsize=10)
        ax[i].set_ylabel('MAPE', fontsize=10)
        ax[i].grid()
        ax[i].tick_params(labelrotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_path}/i1.png')
    plt.close()

    # IHME I1, SEIHRD C1 AND SIRD C1 (val)
    print("IHME I1, SEIR_Testing C1 and SIRD C1 val losses on s2")
    seirt_c1_loss = get_seir_loss_dict(path, 'seirt_c1_loss.csv', num)
    sird_c1_loss = get_seir_loss_dict(path, 'sird_c1_loss.csv', num)
    fig, ax = plt.subplots(4,1, sharex=True, sharey=True, figsize=(20, 15))
    plt.title('MAPE on s2')
    for i, compartment in enumerate(compartments):
        ihme_losses = get_ihme_loss(i1_loss_dict, compartment, 'val', 'mape')
        seirt_losses = get_seir_loss(seirt_c1_loss, compartment, 'val')
        sird_losses = get_seir_loss(sird_c1_loss, compartment, 'val')
        ax[i].plot(dates, ihme_losses, 'o-', label='IHME train MAPE on s2', color='red')
        ax[i].plot(dates, seirt_losses, 'o-', label='SEIRT test MAPE on s2', color='blue')
        ax[i].plot(dates, sird_losses, 'o-', label='SIRD test MAPE on s2', color='green')
        ax[i].title.set_text(compartment)
        ax[i].set_xlabel(f'Training start date (from {start_date})', fontsize=10)
        ax[i].set_ylabel('MAPE', fontsize=10)
        ax[i].grid()
        ax[i].tick_params(labelrotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_path}/ihme_i1_seirt_c1_sird_c1.png')
    plt.close()

    # SEIHRD C1 AND SIRD C1 (val)
    print("SEIR_Testing C1 and SIRD C1 val losses on s3")
    seirt_c3_loss = get_seir_loss_dict(path, 'seirt_baseline_loss.csv', num)
    sird_c3_loss = get_seir_loss_dict(path, 'sird_baseline_loss.csv', num)
    fig, ax = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(20, 15))
    plt.title('MAPE on s3 using baseline model')
    for i, compartment in enumerate(compartments):
        seirt_losses = get_seir_loss(seirt_c3_loss, compartment, 'val')
        sird_losses = get_seir_loss(sird_c3_loss, compartment, 'val')
        ax[i].plot(dates, seirt_losses, 'o-', label='SEIRT test MAPE on s3', color='blue')
        ax[i].plot(dates, sird_losses, 'o-', label='SIRD test MAPE on s3', color='green')
        ax[i].title.set_text(compartment)
        ax[i].set_xlabel(f'Training start date (from {start_date})', fontsize=10)
        ax[i].set_ylabel('MAPE', fontsize=10)
        ax[i].grid()
        ax[i].tick_params(labelrotation=45)
        ax[i].legend()
    plt.tight_layout()
    plt.savefig(f'{output_path}/seirt_c3_sird_c3.png')
    plt.close()

    # Synthetic data experiments with SEIR_Testing
    print("SEIR_Testing C2 on s3")
    seirt_c2_loss = get_seir_loss_dict(path, 'seirt_experiments_loss.csv', num)
    fig, ax = plt.subplots(1,4, sharex=True, sharey=True, figsize=(20,6))
    plt.title('MAPE on s3 using synthetic data')
    for i, compartment in enumerate(compartments):

        colors = ['green', 'red', 'blue']
        labels = ['(using ground truth train data)', '(using IHME forecast as train data)',
                  '(using SEIR forecast as train data)']
        for j in range(3):
            seirt_exp_losses = get_seir_loss_exp(seirt_c2_loss, compartment, 'val', j+1)
            ax[i].plot(dates, seirt_exp_losses, 'o-', label=f'SEIRT test MAPE on s3 {labels[j]}', color=colors[j])
        seirt_baseline_losses = get_seir_loss(seirt_c3_loss, compartment, 'val')
        ax[i].plot(dates, seirt_baseline_losses, 'o-', label='SEIRT baseline test MAPE on s3', color='black')
        ax[i].title.set_text(compartment)
        ax[i].set_xlabel(f'Training start date (from {start_date})', fontsize=10)
        ax[i].set_ylabel('MAPE', fontsize=10)
        ax[i].grid()
        ax[i].tick_params(labelrotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_path}/seirt_experiments.png')
    plt.close()

    # Synthetic data experiments with SIRD
    print("SIRD C2 on s3")
    sird_c2_loss = get_seir_loss_dict(path, 'sird_experiments_loss.csv', num)
    fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(20, 6))
    plt.title('MAPE on s3 using synthetic data')
    for i, compartment in enumerate(compartments):
        colors = ['green', 'red', 'blue']
        labels = ['(using ground truth train data)', '(using IHME forecast as train data)',
                  '(using SIRD forecast as train data)']
        for j in range(3):
            sird_exp_losses = get_seir_loss_exp(sird_c2_loss, compartment, 'val', j + 1)
            ax[i].plot(dates, sird_exp_losses, 'o-', label=f'SIRD test MAPE on s3 {labels[j]}', color=colors[j])
        sird_baseline_losses = get_seir_loss(sird_c3_loss, compartment, 'val')
        ax[i].plot(dates, sird_baseline_losses, 'o-', label='SIRD baseline test MAPE on s3', color='black')
        ax[i].title.set_text(compartment)
        ax[i].set_xlabel(f'Training start date (from {start_date})', fontsize=10)
        ax[i].set_ylabel('MAPE', fontsize=10)
        ax[i].grid()
        ax[i].tick_params(labelrotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_path}/sird_experiments.png')
    plt.close()

    # IHME I1 pointwise
    i1_pointwise_loss_dict = get_ihme_loss_dict(path, 'ihme_i1_pointwise_test_loss.csv', num)
    print("IHME pointwise APE on s2")
    for i, compartment in enumerate(compartments):
        fig, ax = plt.subplots(1, figsize=(20, 10))
        val_losses = get_ihme_pointwise_loss(i1_pointwise_loss_dict, compartment, 'val', 'ape')
        for l in val_losses:
            ax.plot(l, '-o', color='red',
                    label=f'Test ape for {l.index[0]} to {l.index[-1]}')
            ax.title.set_text(compartment)
            ax.set_xlabel('Date', fontsize=10)
            ax.set_ylabel('APE', fontsize=10)
            ax.grid()
            ax.tick_params(labelrotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_path}/ihme_i1_test_pointwise.png')
        plt.close()

    # SEIRT C1 pointwise
    seirt_c1_pointwise_loss_dict = get_seir_pointwise_loss_dict(path, 'seirt_c1_pointwise_val_loss.csv', num)
    print("SEIR_Testing pointwise APE on s2")
    for i, compartment in enumerate(compartments):
        fig, ax = plt.subplots(1, figsize=(20, 10))
        val_losses = get_seir_pointwise_loss(seirt_c1_pointwise_loss_dict, compartment, 'ape')
        for l in val_losses:
            ax.plot(l, '-o', color='blue',
                    label=f'Test ape for {l.index[0]} to {l.index[-1]}')
            ax.title.set_text(compartment)
            ax.set_xlabel('Date', fontsize=10)
            ax.set_ylabel('APE', fontsize=10)
            ax.grid()
            ax.tick_params(labelrotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_path}/seirt_c1_test_pointwise.png')
        plt.close()

    # SIRD C1 pointwise
    sird_c1_pointwise_loss_dict = get_seir_pointwise_loss_dict(path, 'sird_c1_pointwise_val_loss.csv', num)
    print("SIRD pointwise APE on s2")
    for i, compartment in enumerate(compartments):
        fig, ax = plt.subplots(1, figsize=(20, 10))
        val_losses = get_seir_pointwise_loss(sird_c1_pointwise_loss_dict, compartment, 'ape')
        for l in val_losses:
            ax.plot(l, '-o', color='blue',
                    label=f'Test ape for {l.index[0]} to {l.index[-1]}')
            ax.title.set_text(compartment)
            ax.set_xlabel('Date', fontsize=10)
            ax.set_ylabel('APE', fontsize=10)
            ax.grid()
            ax.tick_params(labelrotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_path}/sird_c1_test_pointwise.png')
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", help="folder to results relative to ihme_seir/synth/", required=True)
    parser.add_argument("-n", "--num", help="number of experiments", required=True)
    parser.add_argument("-s", "--shift", help="number of days to shift forward", required=True)
    parser.add_argument("-d", "--start_date", help="first date used", required=True)
    args = parser.parse_args()
    all_plots(args.folder, int(args.num), int(args.shift), args.start_date)
