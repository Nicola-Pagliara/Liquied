import os

import matplotlib.pyplot as plt
import torch
from pytorch_lightning import Trainer
from sklearn.cluster import HDBSCAN

from Dataloader.dataManager import TSDataset
from Models import LTC_AE as model
from utils import Constant as const


def dbscan_encoded(min_cluster=5):
    for time in const.TIME_INTERVAL:
        match time:
            case '15min':
                for nations in const.MIN_15:
                    trainer = Trainer(accelerator='cuda')
                    dataset = TSDataset(batch_size=128,
                                        path_dir=os.path.join(const.TRAIN_15, nations + '_anomaly.json'), timestep=25,
                                        nations=nations)
                    autoencoder = model.LTCAutoEncoder.load_from_checkpoint(
                        os.path.join(const.WEIGHTS_PATH, '15min.ckpt'))
                    encodeds = trainer.predict(autoencoder, dataset)
                    list_encoded = []
                    for encoder in encodeds:
                        encoded = encoder['encoded']
                        list_encoded.append(encoded)

                    # stack expects equal size instead concat accept different size
                    encoded_data = torch.concat(list_encoded, dim=0)
                    hcluster_encoding = HDBSCAN(min_cluster_size=min_cluster).fit_predict(encoded_data)
                    if not os.path.exists(const.SAVE_FIG_EVAL):
                        os.makedirs(const.SAVE_FIG_EVAL)
                    plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=hcluster_encoding, cmap='Paired')
                    plt.savefig(os.path.join(const.SAVE_FIG_EVAL, nations + '_cluster.png'))
                    plt.close()
                    plt.scatter(encoded_data[hcluster_encoding == -1, 0], encoded_data[hcluster_encoding == -1, 1],
                                c='red', s=70, marker='x')
                    plt.scatter(encoded_data[hcluster_encoding != -1, 0], encoded_data[hcluster_encoding != -1, 1],
                                c='blue', s=50)
                    plt.savefig(os.path.join(const.SAVE_FIG_EVAL, nations + '_cluster_anomalies.png'))
                    plt.close()

            case '30min':
                for nations in const.MIN_30:
                    trainer = Trainer(accelerator='cuda')
                    dataset = TSDataset(batch_size=128,
                                        path_dir=os.path.join(const.TRAIN_30, nations + '_anomaly.json'),
                                        timestep=25,
                                        nations=nations)
                    autoencoder = model.LTCAutoEncoder.load_from_checkpoint(
                        os.path.join(const.WEIGHTS_PATH, '30min.ckpt'))
                    encodeds = trainer.predict(autoencoder, dataset)
                    list_encoded = []
                    for encoder in encodeds:
                        encoded = encoder['encoded']
                        list_encoded.append(encoded)

                    # stack expects equal size instead concat accept different size
                    encoded_data = torch.concat(list_encoded, dim=0)
                    hcluster_encoding = HDBSCAN(min_cluster_size=min_cluster).fit_predict(encoded_data)
                    if not os.path.exists(const.SAVE_FIG_EVAL):
                        os.makedirs(const.SAVE_FIG_EVAL)
                    plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=hcluster_encoding, cmap='Paired')
                    plt.savefig(os.path.join(const.SAVE_FIG_EVAL, nations + '_cluster.png'))
                    plt.close()
                    plt.scatter(encoded_data[hcluster_encoding == -1, 0], encoded_data[hcluster_encoding == -1, 1],
                                c='red',
                                s=70, marker='x')
                    plt.scatter(encoded_data[hcluster_encoding != -1, 0], encoded_data[hcluster_encoding != -1, 1],
                                c='blue',
                                s=50)
                    plt.savefig(os.path.join(const.SAVE_FIG_EVAL, nations + '_cluster_anomalies.png'))
                    plt.close()

            case '60min':
                for nations in const.MIN_60:
                    trainer = Trainer(accelerator='cuda')
                    dataset = TSDataset(batch_size=128,
                                        path_dir=os.path.join(const.TRAIN_60, nations + '_anomaly.json'),
                                        timestep=25,
                                        nations=nations)
                    autoencoder = model.LTCAutoEncoder.load_from_checkpoint(
                        os.path.join(const.WEIGHTS_PATH, '15min.ckpt'))
                    encodeds = trainer.predict(autoencoder, dataset)
                    list_encoded = []
                    for encoder in encodeds:
                        encoded = encoder['encoded']
                        list_encoded.append(encoded)

                    # stack expects equal size instead concat accept different size
                    encoded_data = torch.concat(list_encoded, dim=0)
                    hcluster_encoding = HDBSCAN(min_cluster_size=min_cluster).fit_predict(encoded_data)
                    if not os.path.exists(const.SAVE_FIG_EVAL):
                        os.makedirs(const.SAVE_FIG_EVAL)
                    plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=hcluster_encoding, cmap='Paired')
                    plt.savefig(os.path.join(const.SAVE_FIG_EVAL, nations + '_cluster.png'))
                    plt.close()
                    plt.scatter(encoded_data[hcluster_encoding == -1, 0], encoded_data[hcluster_encoding == -1, 1],
                                c='red',
                                s=70, marker='x')
                    plt.scatter(encoded_data[hcluster_encoding != -1, 0], encoded_data[hcluster_encoding != -1, 1],
                                c='blue',
                                s=50)
                    plt.savefig(os.path.join(const.SAVE_FIG_EVAL, nations + '_cluster_anomalies.png'))
                    plt.close()
