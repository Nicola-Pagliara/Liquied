import seaborn as sns
import matplotlib.pyplot as mpt
import matplotlib.dates as mpt_date
import pandas as pd
import numpy as np
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
    log = getLogger("data log")
    dataset = pd.read_excel(original_path, sheet_name=name_data, parse_dates=['Unnamed: 1'])
    data_view = dataset.copy()
    filter_data = data_view.iloc[11:]
    plot_data(data_to_plot=filter_data, flag_EU='')
    clean_dataset(dataset)
    generate_supervised_dataset(filter_data)
    return


def clean_dataset(original_data):
    return


def plot_data(data_to_plot, flag_EU: str):
    mpt.subplots(2, 2, figsize=(20, 5))
    sns.displot(data=data_to_plot[flag_EU])
    sns.boxplot(data=data_to_plot[flag_EU], y=flag_EU)
    sns.histplot(data=data_to_plot[flag_EU])
    filter = data_to_plot[['Unnamed: 1', flag_EU]]
    sns.lineplot(data=filter)
    mpt.gca().xaxis.set_major_formatter(mpt_date.DateFormatter('%Y-%m-%d'))
    mpt.xticks(rotation=45)
    mpt.xlabel('Time')
    mpt.ylabel(f'Load  {flag_EU}')

    return


def generate_supervised_dataset(cleaned_data):
    return


def mad(data):
    series = pd.Series(data)
    median = series.median()
    absolute_deviations = np.abs(series - median)
    mad = absolute_deviations.mean()
    threshold = 3 * mad
    outliers = series[series > threshold]
    return outliers


def IQR(df, col: str):
    Q25 = np.quantile(df[col], 0.25)
    Q75 = np.quantile(df[col], 0.75)
    IQR = Q75 - Q25
    w_range = IQR * 1.5
    # calculating the lower and upper bound value definition of mustache
    w_lower, w_upper = Q25 - w_range, Q75 + w_range
    # Calculating the number of outliers
    out1 = df[df[col] > w_upper]
    out2 = df[df[col] < w_lower]
    return out1, out2


def z_score(data):
    series = pd.Series(data)
    mean = series.mean()
    std_dev = series.std()
    z_scores = (series - mean) / std_dev
    threshold = 3
    outliers = series[z_scores > threshold]
    return outliers
