import json
import asyncio
import websockets
from gnn_rl_agent import GNN_DQNAgent
from sumo_env import SumoEnvironment

# WebSocket config
WS_URI = "ws://localhost:8765"   # replace with your WebSocket server

async def run_agent():
    sumo_cfg = "data/net.sumocfg"
    env = SumoEnvironment(sumo_cfg, sumo_binary="sumo-gui", num_node_features=3, num_actions=3)

    agent = GNN_DQNAgent(num_node_features=3, hidden_dim=32, num_actions=3)
    agent.load("data/gnn_dqn.pth")

    state = env.reset()
    total_reward = 0
    done = False

    async with websockets.connect(WS_URI) as websocket:
        while not done:
            state_x, edge_index = state

            # Get per-node actions (one action per traffic light)
            actions = agent.act(state_x, edge_index, epsilon=0.0)

            # Send actions as JSON over WebSocket
            message = {
                "actions": actions.tolist(),   # convert numpy array to list
                "step": env.step_count
            }
            await websocket.send(json.dumps(message))

            # Step SUMO
            next_state, reward, done = env.step(actions)
            state = next_state
            total_reward += reward

    print(f"✅ Test run finished | Total Reward: {total_reward:.2f}")
    env.close()

# Run the async WebSocket loop
asyncio.run(run_agent())
