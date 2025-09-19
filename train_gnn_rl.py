# import numpy as np
# from gnn_rl_agent import GNN_DQNAgent
# from sumo_env import SumoEnvironment

# # ----------------------------
# # Hyperparameters
# # ----------------------------
# num_node_features = 3   # features per node [halting, count, avg_speed]
# hidden_dim = 32         # hidden dim for GNN
# num_actions = 3         # number of phases per traffic light (adjust to your SUMO net)
# episodes = 10           # number of training episodes
# batch_size = 32
# epsilon = 0.1           # exploration rate

# # Path to SUMO config (adjust path if needed)
# sumo_cfg = "data/net.sumocfg"
# sumo_binary = "sumo"   # use "sumo" for headless fast training

# # ----------------------------
# # Init environment & agent
# # ----------------------------
# env = SumoEnvironment(
#     sumo_cfg=sumo_cfg,
#     sumo_binary=sumo_binary,
#     num_node_features=num_node_features,
#     num_actions=num_actions,
#     max_steps=200,
#     fixed_num_nodes=None,   # auto-detect TLS count
#     topology="fully_connected"
# )

# agent = GNN_DQNAgent(num_node_features, hidden_dim, num_actions)

# # ----------------------------
# # Training Loop
# # ----------------------------
# for ep in range(episodes):
#     state = env.reset()
#     total_reward = 0.0
#     done = False

#     while not done:
#         state_x, edge_index = state

#         # Agent picks one action per node (array of ints)
#         actions = agent.act(state_x, edge_index, epsilon=epsilon)

#         # Environment applies actions
#         next_state, reward, done = env.step(actions)

#         # Store experience
#         agent.remember(state, actions, reward, next_state, done)

#         # Replay & learn
#         agent.replay(batch_size=batch_size)

#         state = next_state
#         total_reward += reward

#     # Update target network after each episode
#     agent.update_target_model()
#     print(f"Episode {ep+1}/{episodes} - Total Reward: {total_reward:.2f}")

# # ----------------------------
# # Save model
# # ----------------------------
# agent.save("data/gnn_dqn.pth")
# env.close()
# print("✅ Training finished and model saved.")

import numpy as np
import matplotlib.pyplot as plt
from gnn_rl_agent import GNN_DQNAgent
from sumo_env import SumoEnvironment

# ----------------------------
# Hyperparameters
# ----------------------------
num_node_features = 3   # features per node [halting, count, avg_speed]
hidden_dim = 64         # bigger hidden dim for stability
num_actions = 3         # number of phases per traffic light (adjust to your SUMO net)
episodes = 200          # more episodes for better learning
batch_size = 64
epsilon = 1.0           # start with full exploration
epsilon_min = 0.05
epsilon_decay = 0.995   # decay rate per episode

# Path to SUMO config (adjust path if needed)
sumo_cfg = "data/net.sumocfg"
sumo_binary = "sumo"   # use "sumo-gui" for visualization, "sumo" for fast training

# ----------------------------
# Init environment & agent
# ----------------------------
env = SumoEnvironment(
    sumo_cfg=sumo_cfg,
    sumo_binary=sumo_binary,
    num_node_features=num_node_features,
    num_actions=num_actions,
    max_steps=500,          # longer episodes
    fixed_num_nodes=None,   # auto-detect TLS count
    topology="fully_connected"
)

agent = GNN_DQNAgent(num_node_features, hidden_dim, num_actions)

# ----------------------------
# Training Loop
# ----------------------------
rewards_per_episode = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        state_x, edge_index = state

        # Agent picks one action per node (array of ints)
        actions = agent.act(state_x, edge_index, epsilon=epsilon)

        # Environment applies actions
        next_state, reward, done = env.step(actions)

        # Store experience
        agent.remember(state, actions, reward, next_state, done)

        # Replay & learn
        agent.replay(batch_size=batch_size)

        state = next_state
        total_reward += reward

    # Update target network after each episode
    agent.update_target_model()
    rewards_per_episode.append(total_reward)

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"Episode {ep+1}/{episodes} - Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

# ----------------------------
# Save model
# ----------------------------
agent.save("data/gnn_dqn.pth")
env.close()
print("✅ Training finished and model saved.")

# ----------------------------
# Plot rewards
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(rewards_per_episode, label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve")
plt.legend()
plt.grid(True)
plt.show()

