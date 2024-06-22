import copy
from init.init import *
from model.model import *


def train(model: nn.Module, dataloaders: dict,
          criterion: nn.Module,
          num_epochs: int, optimizer: optim.Optimizer) -> tuple[nn.Module, list[float], list[float], list[float], list[float]]:
    # train the model
    train_loss = []
    test_loss = []
    train_accuracy_list = []
    val_accuracy_list = []

    best_accuracy = 0
    best_model_wts = None

    for epoch in range(num_epochs):
        model.train()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for features, labels, time_seq_index in dataloaders[phase]:
                features = features.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(features)
                    preds = (outputs > 0.5).float().to(device)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * features.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_accuracy = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'val':
                val_accuracy_list.append(epoch_accuracy)
                test_loss.append(epoch_loss)
                if epoch_accuracy > best_accuracy:
                    best_accuracy = epoch_accuracy
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # save the model
                    save_model(model, LSTM_bin_classification_model_dir, best_accuracy)
            elif phase == 'train':
                train_accuracy_list.append(epoch_accuracy)
                train_loss.append(epoch_loss)
        if (epoch + 1) % 10 == 0:
            print(
                'Epoch [{}/{}], Train Loss: {:.8f}, Test Loss: {:.8f}, Train Accuracy: {:.4f}, Test Accuracy: {:.4f}'.format(
                    epoch + 1, num_epochs, train_loss[-1], test_loss[-1], train_accuracy_list[-1],
                    val_accuracy_list[-1]))

    model.load_state_dict(best_model_wts)

    return model, train_loss, test_loss, train_accuracy_list, val_accuracy_list