import json
import asyncio
import websockets
from gnn_rl_agent import GNN_DQNAgent
from sumo_env import SumoEnvironment

# WebSocket config
WS_HOST = "localhost"
WS_PORT = 8765

async def agent_loop(websocket, path):
    """
    Handles a WebSocket client connection and streams actions each step.
    """
    sumo_cfg = "data/net.sumocfg"
    env = SumoEnvironment(sumo_cfg, sumo_binary="sumo-gui", num_node_features=3, num_actions=3)

    # must match training config
    agent = GNN_DQNAgent(num_node_features=3, hidden_dim=64, num_actions=3)
    agent.load("data/gnn_dqn.pth")

    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        state_x, edge_index = state

        # Agent picks per-node actions (deterministic during test)
        actions = agent.act(state_x, edge_index, epsilon=0.0)

        # Send actions over WebSocket
        message = {
            "actions": actions.tolist(),
            "step": env.step_count
        }
        await websocket.send(json.dumps(message))

        # Step SUMO
        next_state, reward, done = env.step(actions)
        state = next_state
        total_reward += reward

    print(f"✅ Test run finished | Total Reward: {total_reward:.2f}")
    env.close()


async def main():
    async with websockets.serve(agent_loop, WS_HOST, WS_PORT):
        print(f"🚦 WebSocket server started at ws://{WS_HOST}:{WS_PORT}")
        await asyncio.Future()  # run forever until Ctrl+C


if __name__ == "__main__":
    asyncio.run(main())
