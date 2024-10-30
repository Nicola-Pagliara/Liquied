import json

from EDA import EDA as eda
from Train import Train as trn
from Evaluation import Eval as eval


def main():
    trn.train()
    # eda.overview_analysis(original_path='Dataset/time_series.xlsx', name_data=['15min', '30min', '60min'])
    # eval.test_case('Models/autoencoder_weights/epoch=99-step=4500.ckpt')
    # eval.extract_model_anomalies('Models/autoencoder_weights/epoch=99-step=4500.ckpt')


if __name__ == '__main__':
    main()
