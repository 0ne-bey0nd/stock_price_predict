from typing import Any
import numpy as np
from numpy import ndarray
import torch
from torch.utils.data import TensorDataset
import torch.utils.data as data
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_data_label(raw_data, days_seq_len: int, day_after_nums: int) -> pd.DataFrame:
    """

    :param day_after_nums: target after day_nums days to predict
    :param days_seq_len:
    :param raw_data: a pandas.DataFrame
    :return: a pandas.DataFrame
    """
    data_with_label = raw_data.copy()
    # let's get the label first
    # label is binary, if the close price rises, the value is 1, otherwise, the value is 0
    data_with_label.loc[:, 'next_close'] = data_with_label['close'].shift(-day_after_nums)

    # drop the last sample
    # data_with_label = data_with_label.dropna()
    def filter(x):
        if np.isnan(x):
            return np.NAN
        else:
            return 1 if x > 0 else 0

    res = pd.DataFrame((data_with_label['next_close'] - data_with_label['close']).apply(
        filter), columns=['label'])
    if day_after_nums == 0:
        res = res[days_seq_len - 1:]
    else:
        res = res[days_seq_len - 1:-day_after_nums]
    return res


def get_feature_time_seq(raw_data, days_seq_len: int, day_after_nums: int) -> pd.Series:
    """
    get time_seq
    :param raw_data:
    :param days_seq_len:
    :param day_after_nums:
    :return:
    """
    if day_after_nums == 0:
        time_seq = raw_data.index[days_seq_len - 1:]
    else:
        time_seq = raw_data.index[days_seq_len - 1:-day_after_nums]
    return time_seq


def get_data_features_sequence_modeling(raw_data, days_seq_len: int, day_after_nums: int) -> tuple[
    np.ndarray, pd.Series]:
    """
    get data feature by sequence modeling
    :param day_after_nums:
    :param days_seq_len:
    :param raw_data: only features pd.DataFrame
    :return:
    """
    # we need to prepare for the features of every sample
    sample_size = raw_data.shape[0] - days_seq_len - day_after_nums + 1
    # print(sample_size)
    features = []
    for sample_idx in range(0, sample_size):
        sample_features = []
        for day_idx in range(days_seq_len):
            sample_features.append(raw_data.iloc[sample_idx + day_idx, :])
        features.append(sample_features)
    features = np.array(features)
    feature_time_seq = get_feature_time_seq(raw_data, days_seq_len, day_after_nums)
    return features, feature_time_seq


def get_processed_raw_data(raw_data, days_seq_len: int, day_after_nums: int) -> tuple[
    np.ndarray, np.ndarray, pd.Series]:
    """
    get dataset by sequence modeling
    :param days_seq_len:
    :param day_after_nums:
    :param raw_data: only features pd.DataFrame
    :return:
    """
    features, feature_time_seq = get_data_features_sequence_modeling(raw_data, days_seq_len, day_after_nums)
    labels = get_data_label(raw_data, days_seq_len, day_after_nums)
    labels = labels.values
    return features, labels, feature_time_seq


class MyDataset(data.Dataset):
    def __init__(self, features, labels, time_seq):
        self.features = features
        self.labels = labels
        self.time_seq = time_seq

    def __getitem__(self, index):
        return self.features[index], self.labels[index], index

    def get_time_seq(self, index):
        if type(index) is torch.Tensor:
            index = index.tolist()
        return self.time_seq[index]

    def __len__(self):
        return len(self.features)


def get_dataset(raw_data, days_seq_len: int, day_after_nums: int, test_data_ratio: float, scaler: StandardScaler
                ,random_seed:int = 42) -> tuple[MyDataset, MyDataset]:
    """
    get dataset by sequence modeling
    :param random_seed:
    :param test_data_ratio:
    :param scaler:
    :param days_seq_len:
    :param day_after_nums:
    :param raw_data: only features pd.DataFrame
    :return:
    """
    features, labels, time_seq = get_processed_raw_data(raw_data, days_seq_len, day_after_nums)

    dataset = MyDataset(features, labels, time_seq)

    # get train dataset and test dataset
    train_data_size = int((1 - test_data_ratio) * len(dataset))
    test_data_size = len(dataset) - train_data_size
    split_index = train_data_size

    # train_features, test_features, train_labels, test_labels, time_seq_train, time_seq_test = train_test_split(dataset.features, dataset.labels, dataset.time_seq, test_size=test_data_ratio, random_state=random_seed)

    train_features = dataset.features[:split_index]
    test_features = dataset.features[split_index:]
    train_labels = dataset.labels[:split_index]
    test_labels = dataset.labels[split_index:]
    time_seq_train = dataset.time_seq[:split_index]
    time_seq_test = dataset.time_seq[split_index:]


    sequence_len = train_features.shape[1]
    feature_dim = train_features.shape[2]

    # data preprocessing, only normalize the features, not the label
    train_features = scaler.fit_transform(train_features.reshape(-1, feature_dim)).reshape(-1, sequence_len,
                                                                                           feature_dim)
    test_features = scaler.transform(test_features.reshape(-1, feature_dim)).reshape(-1, sequence_len, feature_dim)

    # transform to tensor
    train_features = torch.from_numpy(train_features).float()
    test_features = torch.from_numpy(test_features).float()
    train_labels = torch.from_numpy(train_labels).float()
    test_labels = torch.from_numpy(test_labels).float()

    train_dataset = MyDataset(train_features, train_labels, time_seq_train)
    test_dataset = MyDataset(test_features, test_labels, time_seq_test)

    return train_dataset, test_dataset


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

    # data preprocessing, only normalize the features, not the label
    # print(train_data)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_features = train_data.dataset.tensors[0]
    test_features = test_data.dataset.tensors[0]

    sequence_len = train_features.shape[1]
    feature_dim = train_features.shape[2]

    train_features = scaler.fit_transform(train_features.reshape(-1, feature_dim)).reshape(-1, sequence_len,
                                                                                           feature_dim)
    test_features = scaler.transform(test_features.reshape(-1, feature_dim)).reshape(-1, sequence_len, feature_dim)

    return train_data, test_data


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


def dataset_prepare(raw_data: pd.DataFrame, pred_day_num: int, days_seq_len: int, test_data_ratio: float) -> tuple[
    Any, Any, Any]:
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
