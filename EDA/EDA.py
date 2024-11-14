import datetime
import os.path

import pandas.core.frame
import seaborn as sns
import matplotlib.pyplot as mpt
import pandas as pd
import numpy as np
import json
from utils import Constant as const
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

from utils.Logger import getLogger

"""
Evaluate OOP paradigm class EDA

"""


def overview_analysis():
    """
    This method do a complete overview of times series integrate the following steps:
    1) Extract data
    2) Clean data
    3) format date data
    4) plot data for statistical comparison
    5) generate and save dataset and anomaly data

    :param original_path: time series data path
    :param name_data: sheet name with the execl file to analyze
    :return: path where the preprocessed dataset it's store.
    """
    log = getLogger('EDA Log')
    for sheet in const.TIME_INTERVAL :
        match sheet:
            case '15min':
                log.warn('Enter 15 min case')
                dataset = pd.read_excel(const.ORIGINAL_DATA, sheet_name=sheet, parse_dates=['Unnamed: 1'])
                selected_dataset = dataset.filter(
                    ['AT', 'BE', 'DE', 'DE_50hertz', 'DE_LU', 'DE_amprion', 'DE_tennet', 'HU',
                     'LU', 'NL', 'Unnamed: 1'])
                columns = selected_dataset.columns
                for col in columns:
                    if col == 'Unnamed: 1':
                        continue
                    filter_data = clean_dataset(selected_dataset, col)
                    plot_data(filter_data, col, sheet)
                    generate_anomaly_dataset(filter_data, col, sheet)
                log.warn('Finish 15 min case')
            case '30min':
                log.warn('Enter 30 min case')
                dataset = pd.read_excel(const.ORIGINAL_DATA, sheet_name=sheet, parse_dates=['Unnamed: 1'])
                selected_dataset = dataset.filter(
                    ['CY', 'GB_GBN', 'GB_UKM', 'IE', 'Unnamed: 1'])
                columns = selected_dataset.columns
                for col in columns:
                    if col == 'Unnamed: 1':
                        continue
                    filter_data = clean_dataset(selected_dataset, col)
                    plot_data(filter_data, col, sheet)
                    generate_anomaly_dataset(filter_data, col, sheet)
                log.warn('Finish 30 min case')
            case '60min':
                log.warn('Enter 60 min case')
                dataset = pd.read_excel(const.ORIGINAL_DATA, sheet_name=sheet, parse_dates=['Unnamed: 1'])
                selected_dataset = dataset.filter(
                    ['AT', 'BE', 'DE', 'DE_50hertz', 'DE_LU', 'DE_amprion', 'DE_tennet', 'HU',
                     'LU', 'NL', 'BG', 'CY', 'GB_GBN', 'GB_UKM', 'IE', 'CH', 'CZ', 'DK', 'DK_1', 'DK_2',
                     'EE', 'ES', 'FI', 'FR', 'GR', 'HR', 'IE_sem', 'IT', 'LT', 'LV', 'NO', 'NO_1',
                     'NO_2', 'SK', 'Unnamed: 1'])
                columns = selected_dataset.columns
                for col in columns:
                    if col == 'Unnamed: 1':
                        continue
                    filter_data = clean_dataset(selected_dataset, col)
                    plot_data(filter_data, col, sheet)
                    generate_anomaly_dataset(filter_data, col, sheet)
                log.warn('Finish 60 min case')

    return


def clean_dataset(original_data, col):
    data_to_clean = original_data[['Unnamed: 1', col]].copy()
    clean_data = data_to_clean.dropna(subset=['Unnamed: 1', col])
    return clean_data


def plot_data(data_to_plot: pandas.core.frame.DataFrame, flag_EU: str, name_data: str) -> None:
    """

    :param name_data:
    :param data_to_plot:
    :param flag_EU:
    :return:
    """
    save_path = os.path.join('EDA/Plots', name_data)
    save_path = os.path.join(save_path, flag_EU)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig, ax = mpt.subplots(1, 1, figsize=(10, 8))
    sns.boxplot(data=data_to_plot[[flag_EU]], y=flag_EU, ax=ax)
    ax.set(ylabel=f"Load of {flag_EU}",
           title=f"{flag_EU} Electricity  Consumption BoxPlot ")
    fig.savefig(os.path.join(save_path, 'boxplot.png'))
    mpt.close(fig)
    fig1, ax1 = mpt.subplots(1, 1, figsize=(10, 8))
    sns.histplot(data=data_to_plot[flag_EU], kde=True, ax=ax1)
    ax1.set(title=f"{flag_EU} Electricity  Consumption HistPlot ")
    fig1.savefig(os.path.join(save_path, 'histplot.png'))
    mpt.close(fig1)
    data_to_plot['Unnamed: 1'] = data_to_plot['Unnamed: 1'].astype('datetime64[ns]')
    fig3, ax3 = mpt.subplots(1, 1, figsize=(10, 8))
    sns.lineplot(data=data_to_plot, ax=ax3)
    ax3.set(xlabel="Time Period",
            ylabel=f"Load of {flag_EU}",
            title=f"{flag_EU} Electricity  Consumption  LinePlot \n 2015-2020 ",
            xlim=[datetime.date(2015, 1, 1), datetime.date(2020, 10, 1)])
    fig3.savefig(os.path.join(save_path, 'lineplot.png'))
    mpt.close(fig3)
    return


def generate_anomaly_dataset(cleaned_data, col, name):
    save_path = os.path.join('Train', 'train_data')
    save_path = os.path.join(save_path, name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dict_anomaly = {}
    filt = cleaned_data[col].copy()
    outline_mad = mad(filt)
    outline_iqr = IQR(filt)
    outline_z_score = z_score(filt)
    dict_anomaly.update({f'load of {col}': filt.values.tolist()})
    dict_anomaly.update({f'Mean absolute deviation Anomaly load of {col}': outline_mad.values.tolist()})
    dict_anomaly.update({f'Inter-quartile Range Anomaly of {col}': outline_iqr.values.tolist()})
    dict_anomaly.update({f'Z-score Anomaly of {col}': outline_z_score.values.tolist()})

    with open(save_path + '/' + col + '_anomaly.json', 'w') as f:
        json.dump(dict_anomaly, f)
        f.close()

    return


def mad(data):
    series = pd.Series(data)
    median = series.median()
    absolute_deviations = np.abs(series - median)
    mad = absolute_deviations.mean()
    threshold = 3 * mad
    outliers = series[series > threshold]
    return outliers


def IQR(data):
    Q1 = np.quantile(data, 0.25)
    Q3 = np.quantile(data, 0.75)
    IQR = Q3 - Q1
    w_range = IQR * 1.5
    # calculating the lower and upper bound value definition of mustache
    w_lower, w_upper = Q1 - w_range, Q3 + w_range
    series = pd.Series(data)
    # Calculating outliers
    out1 = series[series > w_upper]
    out2 = series[series < w_lower]
    final_out = pd.concat([out1, out2], axis=0)
    return final_out


def z_score(data):
    series = pd.Series(data)
    mean = series.mean()
    std_dev = series.std()
    z_scores = (series - mean) / std_dev
    threshold = 3
    outliers = series[z_scores > threshold]
    return outliers
