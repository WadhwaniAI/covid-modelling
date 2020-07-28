import matplotlib.pyplot as plt

from viz.utils import axis_formatter


def plot_tests(df_testing):
    fig, axs = plt.subplots(figsize=(18, 12), nrows=2)
    axs[0].plot(df_testing['date'], df_testing['tests'], '--o', color='C0', label='Tests (Actual)')
    axs[0].plot(df_testing['date'], df_testing['tests'].rolling(7, center=True).mean(), '-', color='C0', label='Tests (RA)')
    axs[0].plot(df_testing['date'], df_testing['positives'], '--o', color='orange', label='Positives (Actual)')
    axs[0].plot(df_testing['date'], df_testing['positives'].rolling(7, center=True).mean(), '-', color='orange', label='Positives (RA)')
    
    axis_formatter(axs[0])
    axs[1].plot(df_testing['date'], df_testing['tpr'], '--o', color='red', label='TPR (Actual)')
    axs[1].plot(df_testing['date'], df_testing['tpr'].rolling(7, center=True).mean(), '-', color='red', label='TPR (RA)')
    axis_formatter(axs[1])

    return fig
