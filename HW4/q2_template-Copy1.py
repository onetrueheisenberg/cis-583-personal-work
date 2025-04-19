import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 1. Complete the MLP class
class MLP(nn.Module):
    def __init__(self, n_hidden=1, hidden_size=64, dropout=0., l2_reg=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Dropout(dropout))
        # TODO: Add hidden layers with ReLU activation
        # Add final layer with 1 output and no activation

    def forward(self, x):
        # TODO: Implement forward pass
        pass

# 2. Complete the DeepTripletModel
class DeepTripletModel(nn.Module):
    def __init__(self, n_users, n_items, user_dim=32, item_dim=64, margin=1., 
                 n_hidden=1, hidden_size=64, dropout=0, l2_reg=None):
        super().__init__()
        # TODO: Initialize embedding layers and MLP
        self.margin_loss = MarginLoss(margin)

    def forward(self, inputs):
        # TODO: Implement forward pass
        # Should return margin loss
        pass

# 3. Complete the evaluation function
def average_roc_auc(model, train_data, test_data):
    # TODO: Implement ROC AUC calculation
    # Should return average AUC across all users
    pass

# 4. Implement the training loop
def train_deep_recsys():
    # Load data
    pos_data_train, pos_data_test = load_implicit_data()

    # Initialize models
    hyper_parameters = {
        'user_dim': 32,
        'item_dim': 64,
        'n_hidden': 1,
        'hidden_size': 128,
        'dropout': 0.1,
        'l2_reg': 0.,
        'margin': 1.0
    }

    deep_triplet_model = DeepTripletModel(n_users, n_items, **hyper_parameters)
    deep_match_model = DeepMatchModel(deep_triplet_model.user_layer,
                                      deep_triplet_model.item_layer,
                                      deep_triplet_model.mlp)

    # Training loop
    n_epochs = 20
    for i in range(n_epochs):
        # TODO: Sample triplets and train model
        # TODO: Calculate and print test ROC AUC
        pass