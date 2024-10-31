import numpy
import numpy as np
import os
import datetime
import pandas as pd
from pytorch_lightning import Trainer
from Dataloader.dataManager import TSDataset
from sklearn.metrics import precision_recall_fscore_support
from Models import LTC_AE as model
from utils import Constant as const
from utils import Logger
import json
import torch
import seaborn as sns
from EDA import EDA as eda
import matplotlib.pyplot as mpt


def test_case():
    log = Logger.getLogger('Eval Logger')
    for time in const.TIME_INTERVAL:
        match time:
            case '15min':
                model_trained = model.LTCAutoEncoder.load_from_checkpoint(
                    os.path.join(const.WEIGHTS_PATH, '15min.ckpt'))
                tester = Trainer(accelerator='cuda')
                for nations in const.MIN_15:
                    dataset = TSDataset(batch_size=128,
                                        path_dir=os.path.join(const.TRAIN_15, nations + '_anomaly.json'),
                                        timestep=25, nations=nations)
                    tester.test(model_trained, dataset)
                    log.warn(f'Above test metrics for nation {nations}')
            case '30min':
                model_trained = model.LTCAutoEncoder.load_from_checkpoint(
                    os.path.join(const.WEIGHTS_PATH, '30min.ckpt'))
                tester = Trainer(accelerator='cuda')
                for nations in const.MIN_15:
                    dataset = TSDataset(batch_size=128,
                                        path_dir=os.path.join(const.TRAIN_30, nations + '_anomaly.json'),
                                        timestep=25, nations=nations)
                    tester.test(model_trained, dataset)
                    log.warn(f'Above test metrics for nation {nations}')

            case '60min':
                model_trained = model.LTCAutoEncoder.load_from_checkpoint(
                    os.path.join(const.WEIGHTS_PATH, '60min.ckpt'))
                tester = Trainer(accelerator='cuda')
                for nations in const.MIN_15:
                    dataset = TSDataset(batch_size=128,
                                        path_dir=os.path.join(const.TRAIN_60, nations + '_anomaly.json'),
                                        timestep=25, nations=nations)
                    tester.test(model_trained, dataset)
                    log.warn(f'Above test metrics for nation {nations}')

    return


def extract_model_anomalies():
    model_pred = model.LTCAutoEncoder.load_from_checkpoint(os.path.join(const.WEIGHTS_PATH, '15min.ckpt'))
    for time in const.TIME_INTERVAL:
        match time:
            case '15min':
                for nations in const.MIN_15:
                    inference = Trainer(accelerator='cuda')
                    dataset = TSDataset(batch_size=128,
                                        path_dir=os.path.join(const.TRAIN_15, nations + '_anomaly.json'),
                                        timestep=25, nations=nations)
                    preds = inference.predict(model_pred, dataset, return_predictions=True)
                    target = dataset.predict.norm_data
                    last_pred = preds.pop()
                    last_pred = torch.tensor(last_pred, dtype=torch.float32)
                    pad_tensor = torch.zeros(size=(dataset.batch_size - last_pred.shape[0], last_pred.shape[1],
                                                   last_pred.shape[2]))
                    last_pred = torch.concat([last_pred, pad_tensor])
                    preds.append(last_pred)
                    tensor_preds = torch.stack(preds, dim=1)
                    pad_nu = numpy.asarray(preds).flatten()
                    target = numpy.pad(target, (0, pad_nu.shape[0] - target.shape[0]), 'constant')
                    target = torch.tensor(target, dtype=torch.float32)
                    target = target.reshape(
                        [tensor_preds.shape[0], tensor_preds.shape[1], tensor_preds.shape[2], tensor_preds.shape[3]])
                    anomaly = torch.mean((tensor_preds - target) ** 2, dim=1)
                    # threshold = 2
                    # anomaly = anomaly > threshold
                    anomaly_s = pd.Series(np.asarray(anomaly).flatten())
                    mean, std = dataset.predict.mean, dataset.predict.std
                    denormalized_anomaly = (anomaly_s * std) + mean
                    # anomalous_nt = np.asarray(denormalized_anomaly).astype(int)
                    plot_anomaly(denormalized_anomaly, nations, '15min')
                    # precision, recall, f1_score, _ = precision_recall_fscore_support(binary_labels, anomalous_nt, average='weighted')
                    # print(f'F1_SCORE {f1_score}, precision {precision}, recall {recall}, anomalay = {anomalous_nt}, labels = {binary_labels}')
            case '30min':
                pass

            case '60min':
                pass

    return


def plot_anomaly(anomaly_model, nations, name_sheet):
    with open(os.path.join(const.TRAIN_15, nations + '_anomaly.json')) as file:
        datas = json.load(file)
        file.close()

    dataset = pd.read_excel(const.ORIGINAL_DATA, sheet_name=name_sheet, parse_dates=['Unnamed: 1'])
    clean = eda.clean_dataset(dataset, nations)
    clean['Unnamed: 1'] = clean['Unnamed: 1'].astype('datetime64[ns]')
    outlier_mad = datas['Mean absolute deviation Anomaly load of ' + nations]
    outlier_iqr = datas['Inter-quartile Range Anomaly of ' + nations]
    outlier_z = datas['Z-score Anomaly of ' + nations]
    data = datas['load of ' + nations]
    series_data = pd.Series(data)
    mask_model = series_data.isin(pd.Series(anomaly_model))
    mask_mad = series_data.isin(pd.Series(outlier_mad))
    mask_iqr = series_data.isin(pd.Series(outlier_iqr))
    mask_z = series_data.isin(pd.Series(outlier_z))
    index_mad = series_data.index[mask_mad]
    index_model = series_data.index[mask_model]
    index_iqr = series_data.index[mask_iqr]
    index_z = series_data.index[mask_z]
    anomalies = clean.iloc[index_model]
    anomalies_mad = clean.iloc[index_mad]
    anomalies_iqr = clean.iloc[index_iqr]
    anomalies_z = clean.iloc[index_z]
    anomalies = pd.DataFrame(anomalies).set_index('Unnamed: 1')
    anomalies_m = pd.DataFrame(anomalies_mad).set_index('Unnamed: 1')
    anomalies_i = pd.DataFrame(anomalies_iqr).set_index('Unnamed: 1')
    anomalies_zs = pd.DataFrame(anomalies_z).set_index('Unnamed: 1')
    save_path = os.path.join(const.SAVE_FIG_EVAL, name_sheet)
    final_path = os.path.join(save_path, nations)
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    fig3, ax3 = mpt.subplots(figsize=(10, 8))
    sns.lineplot(data=clean, ax=ax3)
    sns.scatterplot(data=anomalies, palette=['#FF0000'], ax=ax3)
    ax3.set(xlabel="Time Period",
            ylabel=f"Load of {nations}",
            title=f"{nations} Electricity  Consumption  LinePlot with Anomalies \n 2015-2020 ",
            xlim=[datetime.date(2015, 1, 1), datetime.date(2020, 10, 1)])
    mpt.legend(labels=[f' load of {nations}', f'{nations} Anomalies'])
    fig3.savefig(os.path.join(final_path, nations + '_model_anomaly_plot.png'))

    fig4, ax4 = mpt.subplots(figsize=(10, 8))
    sns.lineplot(data=clean, ax=ax4)
    sns.scatterplot(data=anomalies_m, palette=['#FF0000'], ax=ax4)
    ax4.set(xlabel="Time Period",
            ylabel=f"Load of {nations}",
            title=f"{nations} Electricity  Consumption  LinePlot with Anomalies \n 2015-2020",
            xlim=[datetime.date(2015, 1, 1), datetime.date(2020, 10, 1)])
    mpt.legend(labels=[f' load of {nations}', f'{nations} Anomalies'])
    fig4.savefig(os.path.join(final_path, nations + '_mad_anomaly_plot.png'))

    fig5, ax5 = mpt.subplots(figsize=(10, 8))
    sns.lineplot(data=clean, ax=ax5)
    sns.scatterplot(data=anomalies_i, palette=['#FF0000'], ax=ax5)
    ax5.set(xlabel="Time Period",
            ylabel=f"Load of {nations}",
            title=f"{nations} Electricity  Consumption  LinePlot with Anomalies \n 2015-2020 ",
            xlim=[datetime.date(2015, 1, 1), datetime.date(2020, 10, 1)])
    mpt.legend(labels=[f' load of {nations}', f'{nations} Anomalies'])
    fig5.savefig(os.path.join(final_path, nations + '_iqr_anomaly_plot.png'))

    fig6, ax6 = mpt.subplots(figsize=(10, 8))
    sns.lineplot(data=clean, ax=ax6)
    sns.scatterplot(data=anomalies_zs, palette=['#FF0000'], ax=ax6)
    ax3.set(xlabel="Time Period",
            ylabel=f"Load of {nations}",
            title=f"{nations} Electricity  Consumption  LinePlot with Anomalies \n 2015-2020 ",
            xlim=[datetime.date(2015, 1, 1), datetime.date(2020, 10, 1)])
    mpt.legend(labels=[f' load of {nations}', f'{nations} Anomalies'])
    fig6.savefig(os.path.join(final_path, nations + '_zscore_anomaly_plot.png'))

    return


def test_phase():
    test_case()
    extract_model_anomalies()
    return
