import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch_geometric.nn import GCNConv


# ----------------------------
# GNN Q-Network
# ----------------------------
class GNN_QNetwork(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_actions):
        super(GNN_QNetwork, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.mean(x, dim=0)  # Graph-level embedding
        return self.fc(x)         # Q-values for actions


# ----------------------------
# Replay Buffer
# ----------------------------
class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ----------------------------
# DQN Agent with GNN
# ----------------------------
class GNN_DQNAgent:
    def __init__(self, num_node_features, hidden_dim, num_actions, device="cpu"):
        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.device = device

        self.model = GNN_QNetwork(num_node_features, hidden_dim, num_actions).to(self.device)
        self.target_model = GNN_QNetwork(num_node_features, hidden_dim, num_actions).to(self.device)
        self.update_target_model()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer(capacity=5000)
        self.gamma = 0.95

    # ----------------------------
    # Choose action
    # ----------------------------
    def act(self, x, edge_index, epsilon=0.1):
        if np.random.rand() <= epsilon:
            return np.random.randint(self.num_actions)  # exploration
        else:
            x = torch.tensor(x, dtype=torch.float).to(self.device)
            edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)
            q_values = self.model(x, edge_index)
            return torch.argmax(q_values).item()

    # ----------------------------
    # Store transition
    # ----------------------------
    def remember(self, state, action, reward, next_state, done):
        # clamp action to avoid invalid indices
        action = int(np.clip(action, 0, self.num_actions - 1))
        self.memory.push((state, action, reward, next_state, done))

    # ----------------------------
    # Replay and learn
    # ----------------------------
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in batch:
            state_x, state_edge = state
            next_state_x, next_edge = next_state

            state_x = torch.tensor(state_x, dtype=torch.float).to(self.device)
            state_edge = torch.tensor(state_edge, dtype=torch.long).to(self.device)
            next_state_x = torch.tensor(next_state_x, dtype=torch.float).to(self.device)
            next_edge = torch.tensor(next_edge, dtype=torch.long).to(self.device)

            # Current Q-values
            q_values = self.model(state_x, state_edge)

            # Only pick Q-value of the chosen action
            q_value = q_values[action]

            # Target
            if done:
                target_value = torch.tensor(reward, dtype=torch.float, device=self.device)
            else:
                next_q = self.target_model(next_state_x, next_edge).max().item()
                target_value = torch.tensor(reward + self.gamma * next_q, dtype=torch.float, device=self.device)

            # Loss & backprop
            loss = self.criterion(q_value, target_value)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # ----------------------------
    # Sync target network
    # ----------------------------
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # ----------------------------
    # Save & Load
    # ----------------------------
    def save(self, path="data/gnn_dqn.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="data/gnn_dqn.pth"):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.update_target_model()
            print(f"✅ Model loaded from {path}")
