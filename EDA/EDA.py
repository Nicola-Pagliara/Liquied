import seaborn as sns
import matplotlib.pyplot as mpt
import pandas as pd
from utils.Logger import getLogger

"""
Evaluate OOP paradigm class EDA

"""


def overview(original_path: str, name_data: str):
    log = getLogger("data log")
    dataset = pd.read_excel(original_path, sheet_name=name_data, parse_dates=['Unnamed: 1'])
    data_view = dataset.copy()
    filter_data = data_view.iloc[11:]
    sns.displot(filter_data['AT'])
    mpt.xlim(2000, 10500)
    mpt.show()



    return


def minute_analysis():
    return


def hour_analysis():
    return


def day_analysis():
    return


def monthly_analysis():
    return


def yearly_analysis():
    return


def generate_supervised_dataset(original_path, name_data):
    return
