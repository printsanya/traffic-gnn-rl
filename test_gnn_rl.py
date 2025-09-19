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
    Handles a WebSocket client connection and streams actions step-by-step.
    Runs SUMO episodes in an infinite loop until client disconnects.
    """
    sumo_cfg = "data/net.sumocfg"

    # must match training config
    agent = GNN_DQNAgent(num_node_features=3, hidden_dim=64, num_actions=3)
    agent.load("data/gnn_dqn.pth")

    while True:  # keep running episodes until client disconnects
        env = SumoEnvironment(
            sumo_cfg=sumo_cfg,
            sumo_binary="sumo-gui",
            num_node_features=3,
            num_actions=3
        )

        state = env.reset()
        total_reward = 0.0
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
            try:
                await websocket.send(json.dumps(message))
            except websockets.ConnectionClosed:
                print("❌ Client disconnected")
                env.close()
                return

            # Step SUMO
            next_state, reward, done = env.step(actions)
            state = next_state
            total_reward += reward

        print(f"✅ Episode finished | Total Reward: {total_reward:.2f}")
        env.close()


async def main():
    async with websockets.serve(agent_loop, WS_HOST, WS_PORT):
        print(f"🚦 WebSocket server started at ws://{WS_HOST}:{WS_PORT}")
        await asyncio.Future()  # run forever until Ctrl+C


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("🛑 Server stopped by user")
