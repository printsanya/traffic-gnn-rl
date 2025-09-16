import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from collections import deque
import random

# ===============================
# GNN + RL Model
# ===============================
class GNN_QNetwork(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_actions):
        super(GNN_QNetwork, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return x


# ===============================
# DQN Agent with GNN
# ===============================
class GNN_DQNAgent:
    def __init__(self, num_node_features, hidden_dim, num_actions, gamma=0.95, lr=0.001, memory_size=10000):
        self.num_actions = num_actions
        self.gamma = gamma
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = GNN_QNetwork(num_node_features, hidden_dim, num_actions).to(self.device)
        self.target_model = GNN_QNetwork(num_node_features, hidden_dim, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

    def act(self, state_x, state_edge_index):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.num_actions)
        state_x = torch.tensor(state_x, dtype=torch.float32).to(self.device)
        state_edge_index = torch.tensor(state_edge_index, dtype=torch.long).to(self.device)
        q_values = self.model(state_x, state_edge_index)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_x, state_edge_index = state
            next_x, next_edge_index = next_state

            state_x = torch.tensor(state_x, dtype=torch.float32).to(self.device)
            state_edge_index = torch.tensor(state_edge_index, dtype=torch.long).to(self.device)

            next_x = torch.tensor(next_x, dtype=torch.float32).to(self.device)
            next_edge_index = torch.tensor(next_edge_index, dtype=torch.long).to(self.device)

            target = reward
            if not done:
                target += self.gamma * torch.max(self.target_model(next_x, next_edge_index)).item()

            q_values = self.model(state_x, state_edge_index)
            target_f = q_values.clone().detach()
            target_f[action] = target

            loss = nn.MSELoss()(q_values, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # ===============================
    # Save / Load
    # ===============================
    def save(self, path="data/gnn_dqn.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"✅ Model saved to {path}")

    def load(self, path="data/gnn_dqn.pth"):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.update_target_model()
            print(f"✅ Model loaded from {path}")
        else:
            print("⚠️ No saved model found, training from scratch.")
