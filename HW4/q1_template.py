import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Implement the ClassificationModel class
class ClassificationModel(nn.Module):
    def __init__(self, embedding_size, max_user_id, max_item_id):
        super().__init__()
        # TODO: Define your layers here
        # Include user and item embedding layers
        # Add dense layers with appropriate activation
        # Final layer should be Linear(5) followed by softmax in forward

    def forward(self, inputs):
        # TODO: Implement forward pass
        pass

# 2. Complete the training and evaluation code
def train_and_evaluate():
    # Load and preprocess data
    (user_id_train, item_id_train, y_train), (user_id_test, item_id_test, y_test) = load_ml100k()

    # Convert to PyTorch tensors
    user_id_train = torch.tensor(user_id_train, dtype=torch.long)
    item_id_train = torch.tensor(item_id_train, dtype=torch.long)
    y_train = torch.tensor(y_train - 1, dtype=torch.long)  # subtract 1 to make classes 0-4

    user_id_test = torch.tensor(user_id_test, dtype=torch.long)
    item_id_test = torch.tensor(item_id_test, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    max_user_id = max(user_id_train.max(), user_id_test.max()).item()
    max_item_id = max(item_id_train.max(), item_id_test.max()).item()

    # Dataset and DataLoader
    dataset = TensorDataset(user_id_train, item_id_train, y_train)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(TensorDataset(user_id_test, item_id_test, y_test), batch_size=64)

    # Initialize model
    model = ClassificationModel(16, max_user_id, max_item_id)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train model
    train_losses, val_losses = [], []
    for epoch in range(10):
        model.train()
        total_loss = 0
        for user, item, rating in train_loader:
            optimizer.zero_grad()
            output = model((user, item))
            loss = criterion(output, rating)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for user, item, rating in val_loader:
                output = model((user, item))
                loss = criterion(output, rating)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

    # TODO: Plot training and validation loss curves

    # Evaluate model
    model.eval()
    def get_predictions(loader):
        preds, labels = [], []
        with torch.no_grad():
            for user, item, rating in loader:
                output = model((user, item))
                pred = output.argmax(dim=1) + 1  # Add 1 to match original rating scale
                preds.extend(pred.tolist())
                labels.extend(rating.tolist())
        return np.array(preds), np.array(labels)

    train_preds, train_labels = get_predictions(DataLoader(dataset, batch_size=64))
    test_preds, test_labels = get_predictions(test_loader)

    # TODO: Calculate and print MSE and MAE for train and test sets
