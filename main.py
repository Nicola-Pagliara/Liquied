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
    models = model.LTCAutoEncoder(input_dim=1, encoding_dim=16, lr=0.001)
    train = Trainer(accelerator='cuda', max_epochs=100, logger=WandbLogger(name='Liquied', log_model=False),
                    callbacks=[ModelCheckpoint(dirpath='Models/autoencoder_weights')], enable_checkpointing=True)
    dataset = TSDataset(batch_size=128, path_dir='Train/train_data/15min/AT_anomaly.json', timestep=25)
    train.fit(models, dataset)


if __name__ == '__main__':
    main()
