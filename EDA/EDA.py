import seaborn as sns
import matplotlib.pyplot as mpt
import matplotlib.dates as mpt_date
import pandas as pd
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
    # sns.displot(filter_data['AT'])
    # mpt.xlim(2000, 11000)
    # sns.boxplot(filter_data[['AT']], y='AT')
    filter_AT = filter_data[['Unnamed: 1', 'AT']]
    filter_AT['Unnamed: 1'] = pd.to_datetime(filter_AT['Unnamed: 1'])
    filter_AT = filter_AT.sort_values(by='Unnamed: 1')
    sns.lineplot(data=filter_AT)
    mpt.gca().xaxis.set_major_formatter(mpt_date.DateFormatter('%Y-%m-%d'))
    mpt.xticks(rotation=45)
    mpt.xlabel('Time')
    mpt.ylabel('Load AT')
    mpt.show()

    return


def clean_dataset():
    return


def plot_data():
    return


def generate_supervised_dataset(original_path, name_data):
    return
