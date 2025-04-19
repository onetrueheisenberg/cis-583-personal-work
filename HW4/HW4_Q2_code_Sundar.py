#!/usr/bin/env python
# coding: utf-8

# In[48]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split


# In[49]:


# --- Margin Loss ---
class MarginLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, pos_score, neg_score):
        return torch.mean(F.relu(self.margin - pos_score + neg_score))


# In[50]:


# --- MLP Class ---
class MLP(nn.Module):
    def __init__(self, input_dim, n_hidden=1, hidden_size=64, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))

        for _ in range(n_hidden - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))

        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x).squeeze(-1)


# In[51]:


# --- DeepTripletModel ---
class DeepTripletModel(nn.Module):
    def __init__(self, n_users, n_items, user_dim=64, item_dim=64, margin=1., 
                 n_hidden=1, hidden_size=64, dropout=0):
        super().__init__()
        self.user_layer = nn.Embedding(n_users, user_dim)
        self.item_layer = nn.Embedding(n_items, item_dim)
        self.mlp = MLP(input_dim=user_dim, n_hidden=n_hidden, hidden_size=hidden_size, dropout=dropout)
        self.margin_loss = MarginLoss(margin)

    def forward(self, user, item_pos, item_neg):
        user_emb = self.user_layer(user)
        item_pos_emb = self.item_layer(item_pos)
        item_neg_emb = self.item_layer(item_neg)

        pos_score = self.mlp(user_emb * item_pos_emb)
        neg_score = self.mlp(user_emb * item_neg_emb)

        loss = self.margin_loss(pos_score, neg_score)
        return loss


# In[52]:


# --- DeepMatchModel for Evaluation ---
class DeepMatchModel(nn.Module):
    def __init__(self, user_layer, item_layer, mlp):
        super().__init__()
        self.user_layer = user_layer
        self.item_layer = item_layer
        self.mlp = mlp

    def forward(self, user, item):
        user_emb = self.user_layer(user)
        item_emb = self.item_layer(item)
        return self.mlp(user_emb * item_emb)


# In[53]:


# --- ROC AUC Evaluation ---
def average_roc_auc(model, test_data, n_items):
    model.eval()
    scores = []
    for user in test_data:
        pos_items = test_data[user]
        if not pos_items:
            continue

        all_items = list(range(n_items))
        user_tensor = torch.tensor([user] * len(all_items))
        item_tensor = torch.tensor(all_items)

        with torch.no_grad():
            predictions = model(user_tensor, item_tensor).cpu().numpy()

        labels = np.isin(all_items, pos_items).astype(int)
        if len(set(labels)) < 2:
            continue

        auc = roc_auc_score(labels, predictions)
        scores.append(auc)

    return np.mean(scores) if scores else 0


# In[54]:


# --- Load Implicit Data ---
def load_implicit_data():
    df = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    df = df.drop(columns=['timestamp'])
    df['user_id'] -= 1
    df['item_id'] -= 1

    # Train/test split by interaction
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    def build_dict(df):
        data = {}
        for row in df.itertuples():
            data.setdefault(row.user_id, []).append(row.item_id)
        return data

    return build_dict(train_df), build_dict(test_df)


# In[55]:


# --- Train Function ---
def train_deep_recsys():
    train_data, test_data = load_implicit_data()
    n_users = max(max(train_data.keys()), max(test_data.keys())) + 1
    n_items = max(max([max(v) for v in train_data.values()]), max([max(v) for v in test_data.values()])) + 1

    model = DeepTripletModel(
        n_users=n_users,
        n_items=n_items,
        user_dim=64,
        item_dim=64,
        margin=1.0,
        n_hidden=2,
        hidden_size=128,
        dropout=0.2
    )

    match_model = DeepMatchModel(model.user_layer, model.item_layer, model.mlp)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def sample_triplets(data):
        users, pos_items, neg_items = [], [], []
        for user, items in data.items():
            for _ in range(len(items)):
                pos = np.random.choice(items)
                neg = np.random.randint(0, n_items)
                while neg in items:
                    neg = np.random.randint(0, n_items)
                users.append(user)
                pos_items.append(pos)
                neg_items.append(neg)
        return torch.tensor(users), torch.tensor(pos_items), torch.tensor(neg_items)

    for epoch in range(10):
        model.train()
        user, pos, neg = sample_triplets(train_data)
        optimizer.zero_grad()
        loss = model(user, pos, neg)
        loss.backward()
        optimizer.step()

        auc = average_roc_auc(match_model, test_data, n_items)
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, ROC AUC: {auc:.4f}")


# In[56]:


# Run training
train_deep_recsys()

