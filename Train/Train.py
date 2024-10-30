import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from Dataloader.dataManager import TSDataset
from Models import LTC_AE as model
from utils import Constant as const
from utils import Logger


def train():
    log = Logger.getLogger('Train LOGGER')
    count_trained = True
    models = model.LTCAutoEncoder(input_dim=1, encoding_dim=16, lr=1e-3)
    for time in const.TIME_INTERVAL:
        match time:
            case '15min':
                log.warn('Enter train 15min time series')
                trainer = Trainer(accelerator='cuda', max_epochs=100,
                                  logger=WandbLogger(name='Liquied', log_model=False),
                                  callbacks=[ModelCheckpoint(dirpath=const.WEIGHTS_PATH,
                                                             filename='15min'),
                                             EarlyStopping(monitor='val_loss', mode='min', patience=10)],
                                  enable_checkpointing=True)

                for nations in const.MIN_15:
                    if count_trained:
                        dataset = TSDataset(batch_size=128, path_dir=os.path.join(const.TRAIN_15, nations +
                                                                                  '_anomaly.json'), timestep=25,
                                            nations=nations)
                        trainer.fit(models, dataset)
                        count_trained = False
                        log.warn('Complete first train')
                    else:
                        trainer = Trainer(accelerator='cuda', max_epochs=200,
                                          logger=WandbLogger(name='Liquied', log_model=False),
                                          callbacks=[ModelCheckpoint(dirpath=const.WEIGHTS_PATH,
                                                                     filename='15min'),
                                                     EarlyStopping(monitor='val_loss', mode='min', patience=30)],
                                          enable_checkpointing=True)
                        dataset = TSDataset(batch_size=128,
                                            path_dir=os.path.join(const.TRAIN_15, nations +
                                                                  '_anomaly.json'),
                                            timestep=25,
                                            nations=nations)
                        trainer.fit(models, dataset, ckpt_path=os.path.join(const.WEIGHTS_PATH, '15min.ckpt'))

                log.warn(f'Finish train 15 min on  {const.MIN_15} time series, saved weights on {trainer.ckpt_path}')

            case '30min':
                log.warn('Enter train 30min time series')
                count_trained = True
                trainer = Trainer(accelerator='cuda', max_epochs=100,
                                  logger=WandbLogger(name='Liquied', log_model=False),
                                  callbacks=[ModelCheckpoint(dirpath=const.WEIGHTS_PATH,
                                                             filename='30min'),
                                             EarlyStopping(monitor='val_loss', mode='min', patience=10)],
                                  enable_checkpointing=True)

                for nations in const.MIN_30:
                    if count_trained:
                        dataset = TSDataset(batch_size=128,
                                            path_dir=os.path.join(const.TRAIN_30, nations +
                                                                  '_anomaly.json'),
                                            timestep=25,
                                            nations=nations)
                        trainer.fit(models, dataset)
                        count_trained = False
                        log.warn('Complete first train')

                    else:
                        trainer = Trainer(accelerator='cuda', max_epochs=200,
                                          logger=WandbLogger(name='Liquied', log_model=False),
                                          callbacks=[ModelCheckpoint(dirpath=const.WEIGHTS_PATH,
                                                                     filename='30min'),
                                                     EarlyStopping(monitor='val_loss', mode='min', patience=30,
                                                                   verbose=True)],
                                          enable_checkpointing=True)
                        dataset = TSDataset(batch_size=128,
                                            path_dir=os.path.join(const.TRAIN_30, nations +
                                                                  '_anomaly.json'),
                                            timestep=25,
                                            nations=nations)
                        trainer.fit(models, dataset, ckpt_path=os.path.join(const.WEIGHTS_PATH, '30min.ckpt'))

                log.warn(f'Finish train 30 min on  {const.MIN_30} time series, saved weights on {trainer.ckpt_path}')

            case '60min':
                log.warn('Enter train 60min time series')
                count_trained = True
                trainer = Trainer(accelerator='cuda', max_epochs=100,
                                  logger=WandbLogger(name='Liquied', log_model=False),
                                  callbacks=[ModelCheckpoint(dirpath=const.WEIGHTS_PATH,
                                                             filename='60min'),
                                             EarlyStopping(monitor='val_loss', mode='min', patience=20)],
                                  enable_checkpointing=True)

                for nations in const.MIN_60:
                    if count_trained:
                        dataset = TSDataset(batch_size=128,
                                            path_dir=os.path.join(const.TRAIN_60, nations +
                                                                  '_anomaly.json'),
                                            timestep=25,
                                            nations=nations)
                        trainer.fit(models, dataset)
                        count_trained = False
                        log.warn('Complete first train')
                    else:
                        trainer = Trainer(accelerator='cuda', max_epochs=200,
                                          logger=WandbLogger(name='Liquied', log_model=False),
                                          callbacks=[ModelCheckpoint(dirpath=const.WEIGHTS_PATH,
                                                                     filename='60min'),
                                                     EarlyStopping(monitor='val_loss', mode='min', patience=30,
                                                                   verbose=True)],
                                          enable_checkpointing=True)
                        dataset = TSDataset(batch_size=128,
                                            path_dir=os.path.join(const.TRAIN_60, nations +
                                                                  '_anomaly.json'),
                                            timestep=25,
                                            nations=nations)
                        trainer.fit(models, dataset, ckpt_path=os.path.join(const.WEIGHTS_PATH, '60min.ckpt'))

                log.warn(f'Finish train 60 min on  {const.MIN_60} time series, saved weights on {trainer.ckpt_path}')

    return
