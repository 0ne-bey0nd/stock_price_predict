import matplotlib.pyplot as plt
# ========================= utils =========================
def line_chart(data_list: list, label_list: list, title: str = None, xlabel: str = None, ylabel: str = None):
    assert len(data_list) == len(label_list)

    plt.figure(figsize=(20, 8))
    for idx, data in enumerate(data_list):
        plt.plot(data, label=label_list[idx])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()