import os
from typing import Any
import numpy as np
from numpy import ndarray
import torch
from torch.utils.data import TensorDataset
import torch.utils.data as data
import pandas as pd
import torch.nn as nn
import torch.optim as optim
# ========================= model phase =========================
# define the model
class LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, out_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, out_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)

        return out


def get_model(model_name: str, *args, **kwargs) -> nn.Module:
    model_name = model_name.lower()
    if model_name == 'lstm':
        model = LSTM(**kwargs)
    else:
        raise ValueError(f"model_name: {model_name} is not supported")
    return model


def data_set_label(data: pd.DataFrame, pred_day_num: int) -> pd.DataFrame:
    """
    sample the data
    return the dataset to train the model
    :param data:
    :param pred_day_num:
    :return:
    """
    data_with_label = data.copy()
    # let's get the label first
    # label is binary, if the close price rises, the value is 1, otherwise, the value is 0
    data_with_label.loc[:, 'next_close'] = data_with_label['close'].shift(-pred_day_num)
    # drop the last sample
    # data_with_label = data_with_label.dropna()

    data_with_label.loc[:, 'label'] = (data_with_label['next_close'] - data_with_label['close']).apply(
        lambda x: 1 if x > 0 else 0)
    data_with_label = data_with_label.drop('next_close', axis=1)
    return data_with_label


def data_sequence_modeling(data: pd.DataFrame, days_seq_len: int) -> tuple[ndarray, Any, Any]:
    """
    data sequence modeling
    :param days_seq_len:
    :param data:
    :return:
    """
    label = data.loc[:, 'label'][days_seq_len:].values
    label = label.reshape(-1, 1)
    unprocessed_features = data.drop(['label'], axis=1)

    # so let's go on to deal with the data, remember we need our data to become dataset!
    # we already know that our feature is not a matrix like before, it is a 3-D tensor in shape of (sample_size, sequence_len, feature_dim)
    sample_size = label.shape[0]
    # print(sample_size, sequence_len, feature_dim)

    # we need to prepare for the features of every sample
    features = []
    for sample_idx in range(0, sample_size):
        sample_features = []
        for day_idx in range(days_seq_len):
            sample_features.append(unprocessed_features.iloc[sample_idx + day_idx, :])
        features.append(sample_features)
    features = np.array(features)
    time_seq = data.index[days_seq_len:]

    return features, label, time_seq


def data_split_and_preprocessing(features: ndarray, label: ndarray, test_data_ratio: float) -> tuple[Any, Any]:
    """
    data split and preprocessing
    :param test_data_ratio:
    :param features:
    :param label:
    :return:
    """
    dataset = TensorDataset(torch.from_numpy(features).float(), torch.from_numpy(label).float())

    # get train data and test data
    from torch.utils.data import random_split
    train_data_size = int((1 - test_data_ratio) * len(dataset))
    test_data_size = len(dataset) - train_data_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_data_size, test_data_size])

    # data preprocessing, only normalize the features, not the labelprint(train_data)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_features = train_data.dataset.tensors[0]
    test_features = test_data.dataset.tensors[0]

    sequence_len = train_features.shape[1]
    feature_dim = train_features.shape[2]

    train_features = scaler.fit_transform(train_features.reshape(-1, feature_dim)).reshape(-1, sequence_len,
                                                                                           feature_dim)
    test_features = scaler.transform(test_features.reshape(-1, feature_dim)).reshape(-1, sequence_len, feature_dim)

    train_data = TensorDataset(torch.from_numpy(train_features).float(), train_data.dataset.tensors[1])
    test_data = TensorDataset(torch.from_numpy(test_features).float(), test_data.dataset.tensors[1])

    return train_data, test_data


def dataset_prepare(raw_data: pd.DataFrame,pred_day_num:int ,days_seq_len: int, test_data_ratio: float) -> tuple[Any, Any, Any]:
    """
    get the train dataset and test dataset and the time sequence
    :param pred_day_num:
    :param test_data_ratio:
    :param raw_data:
    :param days_seq_len:
    :return:
    """
    data_with_label = data_set_label(raw_data, pred_day_num)
    features, label, time_seq = data_sequence_modeling(data_with_label, days_seq_len)
    train_dataset, test_dataset = data_split_and_preprocessing(features, label, test_data_ratio)
    return train_dataset, test_dataset, time_seq


# save the model
def save_model(model: torch.nn.Module, model_dir: str, model_acc: float):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = 'LSTM_bin_classification.pth'
    model_path = os.path.join(model_dir, model_name)
    # delete the previous model
    if os.path.exists(model_path):
        os.remove(model_path)
    torch.save(model.state_dict(), model_path)
    # save the accuracy
    with open(os.path.join(model_dir, 'accuracy.txt'), 'w') as f:
        f.write(str(model_acc))

    print('Model saved at {}'.format(model_path), 'Accuracy:', model_acc)


def train(model: nn.Module, train_loader: data.DataLoader, test_loader: data.DataLoader, criterion: nn.Module,
          num_epochs: int, optimizer: optim.Optimizer) -> tuple[
    nn.Module, list[float], list[float], list[float], list[float]]:
    # train the model
    train_loss = []
    test_loss = []
    train_accuracy_list = []
    test_accuracy_list = []
    best_accuracy = 0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0
        train_accuracy = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            train_loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicted = (outputs > 0.5).float()
            # print(predicted.shape, labels.shape)
            train_accuracy += (predicted == labels).sum()
            train_accuracy = train_accuracy.item()
        # print(train_accuracy)
        # print(train_accuracy/len(train_data))
        train_loss.append(train_loss_sum / len(train_loader))
        train_accuracy_list.append(train_accuracy / len(train_loader.dataset))

        model.eval()
        test_loss_sum = 0
        test_accuracy = 0
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)
                test_loss_sum += loss.item()

                predicted = (outputs > 0.5).float()
                test_accuracy += (predicted == labels).sum()
                test_accuracy = test_accuracy.item()
        test_loss.append(test_loss_sum / len(test_loader))
        test_accuracy_list.append(test_accuracy / len(test_loader.dataset))
        if test_accuracy_list[-1] > best_accuracy:
            best_accuracy = test_accuracy_list[-1]
            best_model = model
            # save the model
            save_model(model, LSTM_bin_classification_model_dir, best_accuracy)

        if (epoch + 1) % 10 == 0:
            print(
                'Epoch [{}/{}], Train Loss: {:.8f}, Test Loss: {:.8f}, Train Accuracy: {:.4f}, Test Accuracy: {:.4f}'.format(
                    epoch + 1, num_epochs, train_loss[-1], test_loss[-1], train_accuracy_list[-1],
                    test_accuracy_list[-1]))

    return best_model, train_loss, test_loss, train_accuracy_list, test_accuracy_list