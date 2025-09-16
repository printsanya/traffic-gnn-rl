# GNN + RL Traffic Light Control (Centralized Agent)

This repo contains the **GNN + RL agent** that decides traffic light phases for multiple intersections.  
SUMO integration is expected from a teammate.

---

## 📂 Files
- `gnn_rl_agent.py` → Core model & DQN agent with GNN.
- `train_gnn_rl.py` → Training loop (currently uses random data, replace with SUMO).
- `test_gnn_rl.py` → Loads trained model and runs inference.
- `data/gnn_dqn.pth` → Saved model file.

---

## 🔹 Where SUMO fits in
- Replace the **dummy state_x and reward** in `train_gnn_rl.py` with:
  - `state_x`: node features for each intersection (`queue_length, avg_wait_time, occupancy` etc.)
  - `edge_index`: graph of connected intersections.
  - `reward`: e.g., negative total waiting time or queue length.
- Agent output (`action`) = traffic light phase to apply.

---

## 🔹 Training
```bash
python train_gnn_rl.py
