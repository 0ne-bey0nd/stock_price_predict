from init import *
from data.data import *
from model.model import *
from utils.utils import *

# ========================= run =========================
# ========================= data phase =========================
stock_code = 'sz399300'
day_nums = 10000
raw_data = get_data(stock_code, day_nums)
raw_data.shape

# ========================= model phase =========================

days_seq_len = 7
test_data_ratio = 0.3
pred_day_num = 1
train_dataset, test_dataset, time_seq = dataset_prepare(raw_data,pred_day_num ,days_seq_len, test_data_ratio)
sample_size, sequence_len, feature_dim = train_dataset.tensors[0].shape
print(f"sample_size: {sample_size}, sequence_len: {sequence_len}, feature_dim: {feature_dim}")


input_dim = feature_dim
hidden_dim = 16
num_layers = 2
out_dim = 1
batch_size = 32
learning_rate = 0.01
num_epochs = 500

# prepare the data loader
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = LSTM(input_dim, hidden_dim, num_layers, out_dim).to(device)
print(model)

# define the loss function and the optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train the model
best_model, train_loss, test_loss, train_accuracy_list, test_accuracy_list = train(model, train_loader, test_loader,
                                                                                   criterion,
                                                                                   num_epochs,optimizer)

line_chart([train_loss, test_loss], ['train_loss', 'test_loss'], '')
plt.savefig(os.path.join(images_dir, 'loss.png'))

line_chart([train_accuracy_list, test_accuracy_list], ['train_accuracy', 'test_accuracy'], '')
plt.savefig(os.path.join(images_dir, 'accuracy.png'))
