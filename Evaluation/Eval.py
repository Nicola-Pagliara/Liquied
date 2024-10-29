import numpy
import numpy as np
from pytorch_lightning import Trainer
from Dataloader.dataManager import TSDataset
from Models import LTC_AE as model
import torch


def test_case(path_weights):
    model_trained = model.LTCAutoEncoder.load_from_checkpoint(path_weights)
    tester = Trainer(accelerator='cuda')
    dataset = TSDataset(batch_size=128, path_dir='Train/train_data/15min/LU_anomaly.json', timestep=25)
    tester.test(model_trained, dataset)
    return


def extract_model_anomalies(path_weights):
    model_pred = model.LTCAutoEncoder.load_from_checkpoint(path_weights)
    inference = Trainer(accelerator='cuda')
    dataset = TSDataset(batch_size=128, path_dir='Train/train_data/15min/AT_anomaly.json', timestep=25)
    preds = inference.predict(model_pred, dataset, return_predictions=True)
    target = dataset.predict.norm_data
    last_pred = preds.pop()
    last_pred = torch.tensor(last_pred, dtype=torch.float32)
    pad_tensor = torch.zeros(size=(dataset.batch_size - last_pred.shape[0], last_pred.shape[1], last_pred.shape[2]))
    last_pred = torch.concat([last_pred, pad_tensor])
    preds.append(last_pred)
    tensor_preds = torch.stack(preds, dim=1)
    pad_nu = numpy.asarray(preds).flatten()
    target = numpy.pad(target, (0, pad_nu.shape[0] - target.shape[0]), 'constant')
    target = torch.tensor(target, dtype=torch.float32)
    target = target.reshape(
        [tensor_preds.shape[0], tensor_preds.shape[1], tensor_preds.shape[2], tensor_preds.shape[3]])
    anomaly = torch.mean((tensor_preds - target) ** 2, dim=1)
    threshold = 3
    threshold_2 = anomaly.mean() + 2 * anomaly.std()
    # anomaly = np.asarray(anomaly)
    anomalous = anomaly > threshold
    anomalous_2 = anomaly > threshold_2
    mean, std = dataset.predict.mean, dataset.predict.std
    denormalized_anomaly = (anomalous_2 * std) + mean
    print(f'{denormalized_anomaly}')

    return
