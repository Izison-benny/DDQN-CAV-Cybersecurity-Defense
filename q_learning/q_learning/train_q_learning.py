from pathlib import Path

train_q_learning_code = '''\
import gym
import numpy as np
import pandas as pd
import os
from q_learning_agent import QLearningAgent
from cav_env_v3 import CAVExecutionEnv

# ========================== CONFIG ==========================
EPISODES = 3000
MAX_STEPS = 300
os.makedirs("results", exist_ok=True)

# ========================== ENV INIT ==========================
env = CAVExecutionEnv(max_steps=MAX_STEPS)
state_size = env.observation_space.shape[0]
action_shape = env.action_space.nvec  # MultiDiscrete([2,2,3])

# Flatten MultiDiscrete to discrete action indices
from itertools import product
action_map = list(product(*[range(n) for n in action_shape]))
action_size = len(action_map)

agent = QLearningAgent(env, state_size, action_size)

# ========================== LOGGING SETUP ==========================
logs = []
reward_log = []
switch_log = []
qos_log = []
zero_day_log = []  
security_level_log = []

total_steps = 0
total_intrusions = 0
total_zero_days = 0
total_attack_attempts = 0
total_reward_all = 0

# ========================== TRAINING LOOP ==========================
print(f" Starting Q-Learning training for {EPISODES} episodes...")
print(f"Observation space: {state_size} | Action space: {action_shape} → {action_size} flattened actions")

for episode in range(1, EPISODES + 1):
    state = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        action_index = agent.act(state)
        action = list(action_map[action_index])
        next_state, reward, done, info = env.step(action)
        agent.learn(state, action_index, reward, next_state, done)
        state = next_state

        total_reward += reward
        step += 1
        total_steps += 1

        if info.get("attack_status") == 1:
            total_intrusions += 1
            if info.get("attack_type") == "zero-day":
                total_zero_days += 1

        #  Log 1 or 0 for each step to track zero-day per step
        zero_day_log.append(1 if info.get("attack_type") == "zero-day" else 0)

        if info.get("attack_status") in [0, 1]:
            total_attack_attempts += 1

    reward_log.append(total_reward)
    qos_log.append(info.get("qos", 0.0))
    switch_log.append(info.get("switches", 0))
    security_level_log.append(info.get("security_level", "UNKNOWN"))
    total_reward_all += total_reward

    avg_reward = total_reward_all / episode

    #  Compute per-episode zero-days correctly
    episode_zero_days = sum(zero_day_log[-step:])

    logs.append({
        "Episode": episode,
        "Reward": total_reward,
        "CumulativeAvg": avg_reward,
        "Security": 100.0 * (1 - (total_intrusions / max(1, total_attack_attempts))),
        "QoS": info.get("qos", 0.0) * 100,
        "Switches": info.get("switches", 0),
        "zero_day_intrusions": episode_zero_days,
        "SecurityLevel": info.get("security_level", "UNKNOWN")
    })

    if episode % 100 == 0:
        print(f"Episode {episode}/{EPISODES} | "
              f"Reward: {total_reward:.2f} | "
              f"Cumulative Avg: {avg_reward:.2f} | "
              f"Security: {logs[-1]['Security']:.2f}% | "
              f"QoS: {logs[-1]['QoS']:.2f}% | "
              f"Switches: {logs[-1]['Switches']:.1f} | "
              f"Security Level: {logs[-1]['SecurityLevel']}")

# ==================== FINAL METRICS ====================
df = pd.DataFrame(logs)
df.to_csv("results/q_learning_metrics.csv", index=False)

final = df.iloc[-1]
intrusion_rate = 100 - final['Security']

avg_qos = np.mean(qos_log)
avg_switches = np.mean(switch_log)
security_rate = (total_attack_attempts - total_intrusions) / total_attack_attempts if total_attack_attempts > 0 else 1.0
zday_rate = total_zero_days / total_intrusions if total_intrusions > 0 else 0
mtbi = total_steps / total_intrusions if total_intrusions > 0 else float('inf')
avg_security_level = max(set(security_level_log), key=security_level_log.count)

print("\\n" + "="*30 + " FINAL PERFORMANCE " + "="*30)
print(f"→ Total Reward:         {sum(reward_log):>10.2f}")
print(f"→ Security Rate:        {security_rate:>10.2%} ({total_attack_attempts} attacks)")
print(f"→ QoS Uptime:           {avg_qos:>10.2%}")
print(f"→ Intrusions:           {total_intrusions:>10} ({(total_intrusions/total_steps):.2%})")
print(f"→ Zero-Day Intrusions:  {total_zero_days:>10} ({zday_rate:.1%})")
print(f"→ Defended Attacks:     {total_attack_attempts - total_intrusions:>10}")
print(f"→ MTBI:                 {mtbi:>10.1f} steps")
print(f"→ Avg Switches:         {avg_switches:>10.1f}/episode")
print(f"→ Final Epsilon:        {agent.epsilon:>10.4f}")
print(f"→ Security Level:       {avg_security_level:>10}")
print(f"→ Total Steps:          {total_steps:>10}")
print("="*80)
print(" Q-Learning metrics saved to results/q_learning_metrics.csv")
'''

# Save to file
Path("train_q_learning.py").write_text(train_q_learning_code)
print(" train_q_learning.py saved successfully!")
