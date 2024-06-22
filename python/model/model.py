from init.init import *
import torch.nn.init as init
# 初始化LSTM层的函数
def init_lstm_weights(layer):
    if isinstance(layer, nn.LSTM):
        for name, param in layer.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

# 初始化线性层的函数
def init_linear_weights(layer):
    if isinstance(layer, nn.Linear):
        init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            layer.bias.data.fill_(0)
# ========================= model phase =========================
# define the model
class LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, out_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Sequential(torch.nn.Linear(hidden_dim, out_dim),
                                     )

        self.sigmoid = torch.nn.Sigmoid()

        self.apply(init_lstm_weights)
        self.apply(init_linear_weights)

    def forward(self, x, state=None):
        h0 = self.begin_state(x.size(0)).to(get_torch_device())
        c0 = self.begin_state(x.size(0)).to(get_torch_device())

        out, (hn, cn) = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

    def begin_state(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)


def get_model(model_name: str, *args, **kwargs) -> nn.Module:
    model_name = model_name.lower()
    if model_name == 'lstm':
        model = LSTM(**kwargs)
    else:
        raise ValueError(f"model_name: {model_name} is not supported")
    return model


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
