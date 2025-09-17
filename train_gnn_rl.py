import numpy as np
import torch
from gnn_rl_agent import GNN_DQNAgent

num_nodes = 4
num_node_features = 3
num_actions = 3
hidden_dim = 16

agent = GNN_DQNAgent(num_node_features, hidden_dim, num_actions)

# Edge index must be shape [2, num_edges]
edge_index = torch.tensor([
    [0, 1, 2, 3, 0, 2],
    [1, 0, 3, 2, 2, 0]
], dtype=torch.long)

episodes = 5
for ep in range(episodes):
    state_x = np.random.rand(num_nodes, num_node_features)
    state = (state_x, edge_index)
    total_reward = 0

    for t in range(10):
        action = agent.act(state_x, edge_index)

        next_state_x = np.random.rand(num_nodes, num_node_features)
        reward = np.random.randn()
        done = (t == 9)

        next_state = (next_state_x, edge_index)
        agent.remember(state, action, reward, next_state, done)
        agent.replay(batch_size=16)
        state = next_state
        total_reward += reward

    agent.update_target_model()
    print(f"Episode {ep+1} - Total Reward: {total_reward:.2f}")

agent.save("data/gnn_dqn.pth")
