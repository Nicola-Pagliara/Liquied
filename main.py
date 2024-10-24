import json

from EDA import EDA as eda
from Models import LTC_AE as model
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from Dataloader.dataManager import TSDataset


def main():
    """

    :return:
    """

    # eda.overview_analysis(original_path='Dataset/time_series.xlsx', name_data=['15min', '30min', '60min'])
    models = model.LTC_AutoEncoder(input_dim=1, encoding_dim=1, lr=0.001)
    train = Trainer(accelerator='cpu', max_epochs=1, logger=WandbLogger(name='Liquied', log_model='all'),
                    callbacks=[ModelCheckpoint(dirpath='Models/autoencoder_weights')], enable_checkpointing=True)
    dataset = TSDataset(batch_size=1, path_dir='Train/train_data/15min/AT_anomaly.json')
    train.fit(models, dataset)


if __name__ == '__main__':
    main()
