# DDQN-CAV-Cybersecurity-Defense

This repository provides a Double Deep Q-Network (DDQN) based adaptive cybersecurity defense system for Connected and Autonomous Vehicles (CAVs). The system dynamically switches between module-language configurations to minimize attack risks and maximize Quality of Service (QoS) under cyber-threat scenarios.

---

## 📁 Project Structure

| File/Folder               | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `ddqn_agent.py`           | Implements the DDQNAgent with replay memory, target network, and epsilon-greedy policy. |
| `train_ddqn.py`           | DDQN training script for the CAV environment (`cav_env_v3`). Saves metrics and model. |
| `cav_env_v3.py`           | Custom OpenAI Gym environment modeling modular software with CVSS-based risk profiles. |
| `q_learning/`             | Folder containing Q-Learning-based baseline implementation.                |
| ├── `q_learning_agent.py` | Lightweight tabular Q-Learning agent adapted for `cav_env_v3`.              |
| ├── `train_q_learning.py` | Training script for the Q-Learning agent using the same environment and metrics. |
| └── `ql_defense.py`       | Earlier heuristic Q-Learning-based defense prototype (legacy comparison).   |
| `results/`                | Folder where training metrics and logs are stored after execution.         |
| `LICENSE`                 | MIT License for open-source reuse.                                         |

---

##  How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt


