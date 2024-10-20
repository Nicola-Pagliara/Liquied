import datetime
import os.path

import seaborn as sns
import matplotlib.pyplot as mpt
import pandas as pd
import numpy as np
import json
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

from utils.Logger import getLogger

"""
Evaluate OOP paradigm class EDA

"""


def overview_analysis(original_path: str, name_data: str):
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
    dataset = pd.read_excel(original_path, sheet_name=name_data, parse_dates=['Unnamed: 1'])
    filter_dataset = clean_dataset(dataset, 'AT')
    plot_data(data_to_plot=filter_dataset, flag_EU='AT')
    # generate_anomaly_dataset(filter_data, 'Train/train_data', 'AT')
    data_view = dataset.copy()
    filter_data = data_view.iloc[11:]
    # plot_data(data_to_plot=filter_data, flag_EU='')
    # clean_dataset(dataset)
    generate_anomaly_dataset(filter_data, 'Train/train_data', 'AT')
    return


def clean_dataset(original_data, col):
    data_to_clean = original_data[['Unnamed: 1', col]].copy()
    clean_data = data_to_clean.dropna(subset=['Unnamed: 1', col])
    return clean_data


def plot_data(data_to_plot, flag_EU: str):
    """

    :param data_to_plot:
    :param flag_EU:
    :return:
    """
    """
    sns.boxplot(data=data_to_plot[[flag_EU]], y=flag_EU)
    mpt.show()
    mpt.close()
    sns.histplot(data=data_to_plot[flag_EU], kde=True)
    mpt.show()
    mpt.close()
    """
    log = getLogger('Plot data logger')
    data_to_plot['Unnamed: 1'] = data_to_plot['Unnamed: 1'].astype('datetime64[ns]')
    fig, ax = mpt.subplots(1, 1, figsize=(10, 8))
    sns.lineplot(data=data_to_plot, ax=ax)
    ax.set(xlabel="Time Period",
           ylabel=f"Load of {flag_EU}",
           title=f"{flag_EU} Electricity  Consumption \n 2015-2020 ",
           xlim=[datetime.date(2015, 1, 1), datetime.date(2020, 10, 1)])

    mpt.show()

    return


def generate_anomaly_dataset(cleaned_data, save_path, col):
    dict_anomaly = {}
    filt = cleaned_data[col].copy()
    outline_mad = mad(filt)
    outline_iqr = IQR(filt)
    outline_z_score = z_score(filt)
    # dict_anomaly.update({f'load of {col}': filt.values.tolist()})
    dict_anomaly.update({f'Mean absolute deviation Anomaly load of {col}': outline_mad.values.tolist()})
    dict_anomaly.update({f'Interquartile Range Anomaly of {col}': outline_iqr.values.tolist()})
    dict_anomaly.update({f'Z-score Anomaly of {col}': outline_z_score.values.tolist()})
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path + '/' + col + 'anomaly.json', 'w') as f:
        json.dump(dict_anomaly, f)

    outline_mad = mad(cleaned_data[col])
    outline_iqr = IQR(cleaned_data, col)
    outline_z_score = z_score(cleaned_data[col])
    dict_anomaly.update({f'load of {col}': cleaned_data.values})
    dict_anomaly.update({f'Mean absolute deviation Anomaly load of {col}': outline_mad.values})
    dict_anomaly.update({f'Interquartile Range upper Anomaly of {col}': outline_iqr[0]})
    dict_anomaly.update({f'Z-score Anomaly of {col}': outline_z_score.values})
    if os.path.exists(save_path):
        os.removedirs(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path + '/' + col + '.json', 'w') as f:
        json.dump(dict_anomaly, f)
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
