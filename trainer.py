import os
import gc
import random
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, matthews_corrcoef, classification_report, f1_score, recall_score, precision_score, accuracy_score,  roc_auc_score, average_precision_score


def test_4predict_inbatch(model, features, device, batch_size=32):
    model.eval()
    dataset = TensorDataset(features)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            outputs = model(x).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()
    return all_preds, all_probs


def loadmodel_4predict(model_dir, model_name, x_test, device):
    model = torch.load(os.path.join(model_dir, model_name), map_location=device)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
        model.eval()
    _, test_probs = test_4predict_inbatch(model, x_test, device, batch_size=128)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return test_probs

def loadmodel_4test(model_dir, model_name, x_test, device):
    model = torch.load(os.path.join(model_dir, model_name), map_location=device)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
        model.eval()
    test_preds, test_probs = test_4predict_inbatch(model, x_test, device, batch_size=128)
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return test_preds, test_probs
