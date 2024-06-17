from .Ashare import *
import pandas as pd
# ========================= data phase =========================

def produce_data(stock_code: str, day_nums: int) -> pd.DataFrame:
    """
    get raw data from internet or somewhere, just get , not handle the data
    :param stock_code:
    :param day_nums:
    :return:
    """
    return get_price(stock_code, frequency='1d', count=day_nums)


def process_data(stock_code: str, day_nums: int) -> pd.DataFrame:
    """
    handle the raw data, not for the train phase
    :param stock_code:
    :param day_nums:
    :return:
    """
    data = produce_data(stock_code, day_nums)

    # drop null sample
    data = data.dropna()
    # drop the rows containing the 0
    data = data[~(data == 0).any(axis=1)]

    # get data
    # print(data.shape)

    # we define the problem that we predict close price of a day use the previous days_seq_len day's data
    return data


def get_data(stock_code: str, day_nums: int) -> pd.DataFrame:
    """
    the interface to get the preliminary data
    :param stock_code:
    :param day_nums:
    :return:
    """
    return process_data(stock_code, day_nums)
