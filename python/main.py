from init_env import *
from sklearn.preprocessing import StandardScaler

# hyper parameters
stock_code = 'sz399300'
day_nums = 10000
random_seed = 42
day_after_nums = 1  # target after day_after_nums days to predict
days_seq_len = 7
test_data_ratio = 0.3

hidden_dim = 16
num_layers = 1
out_dim = 1
batch_size = 32
learning_rate = 0.01
num_epochs = 500

# ========================= run =========================
# ========================= data collect phase =========================
raw_data = get_data(stock_code, day_nums)
print(raw_data.shape)
# ========================= dataset prepare (data modeling) phase =========================

# feature scaling
normalizer = StandardScaler()
train_dataset, test_dataset = get_dataset(raw_data, days_seq_len, day_after_nums, test_data_ratio, normalizer)
print(train_dataset, test_dataset)
sample_size, sequence_len, feature_dim = train_dataset.features.shape
print(f"sample_size: {sample_size}, sequence_len: {sequence_len}, feature_dim: {feature_dim}")
print(train_dataset.time_seq[:10])

# prepare the data loader
generator = torch.Generator().manual_seed(random_seed)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ========================= model train phase =========================
input_dim = feature_dim

model = LSTM(input_dim, hidden_dim, num_layers, out_dim).to(device)
print(model)

# define the loss function and the optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train the model
best_model, train_loss, test_loss, train_accuracy_list, test_accuracy_list = train(model, {'train': train_loader,
                                                                                           'val': test_loader},
                                                                                   criterion, num_epochs, optimizer)

line_chart([train_loss, test_loss], ['train_loss', 'test_loss'], '')
plt.savefig(os.path.join(images_dir, 'loss.png'))

line_chart([train_accuracy_list, test_accuracy_list], ['train_accuracy', 'test_accuracy'], '')
plt.savefig(os.path.join(images_dir, 'accuracy.png'))
