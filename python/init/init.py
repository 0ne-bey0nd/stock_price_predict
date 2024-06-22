# ========================= init phase =========================
import os
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_torch_device():
    return device


print(f"torch version: {torch.__version__} device: {device}")
project_dir = pathlib.Path('.').absolute().parent

models_dir = os.path.join(project_dir, 'models')
images_dir = os.path.join(project_dir, 'images')


def make_sure_dir_exists(dir):
    os.makedirs(dir, exist_ok=True)


make_sure_dir_exists(models_dir)
make_sure_dir_exists(images_dir)
LSTM_bin_classification_model_dir = os.path.join(models_dir, 'LSTM_bin_classification')

print(f"model_dir: {models_dir}")
print(f"images_dir: {images_dir}")
