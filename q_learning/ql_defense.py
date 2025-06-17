# q_learning_defense.py

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ===============================
# Configuration and Setup
# ===============================
services = ["Python", "C++", "Java"]

baseline_vulnerabilities = {
    "Python": {"insecure_deserialization": 0.5, "code_injection": 0.6, "sql_injection": 0.4},
    "Java":   {"xml_external_entities": 0.3, "code_injection": 0.2, "sql_injection": 0.3},
    "C++":     {"buffer_overflow": 0.8, "code_injection": 0.7, "memory_corruption": 0.6},
}

severity_weights = {
    "insecure_deserialization": 0.7, "code_injection": 0.9, "sql_injection": 0.8,
    "xml_external_entities": 0.6, "buffer_overflow": 1.0, "memory_corruption": 0.9,
}

switching_costs = {
    "Python": {"Python": 0, "C++": 7, "Java": 5},
    "C++": {"Python": 6, "C++": 0, "Java": 8},
    "Java": {"Python": 4, "C++": 6, "Java": 0},
}

# Q-Learning Hyperparameters
EPSILON = 0.1
MIN_EPSILON = 0.05
LEARNING_RATE = 0.05
DISCOUNT_FACTOR = 0.95
EPOCHS = 1000
MAX_STEPS_PER_EPOCH = 50
EPSILON_DECAY = lambda epoch: max(MIN_EPSILON, EPSILON * (0.995 ** (epoch / 100)))

# Initialize structures
Q_table = np.zeros((len(services), len(services)))
service_frequency = {service: 0 for service in services}
vulnerability_levels = {service: vulns.copy() for service, vulns in baseline_vulnerabilities.items()}
vulnerability_history = {service: [] for service in services}

# ===============================
# Simulation Functions
# ===============================
def update_vulnerabilities():
    for service, vul_data in vulnerability_levels.items():
        for vul_type in vul_data:
            change = random.uniform(-0.2, 0.2)
            vul_data[vul_type] = max(0.1, min(1.0, vul_data[vul_type] + change))
        avg = sum(vul_data.values()) / len(vul_data)
        vulnerability_history[service].append(avg)


def hackers_action():
    for _ in range(2):
        target = random.choice(services)
        vul_type = random.choice(list(vulnerability_levels[target].keys()))
        increase = random.uniform(0.3, 0.5)
        vulnerability_levels[target][vul_type] = min(1.0, vulnerability_levels[target][vul_type] + increase)


def get_reward(current, next_):
    curr_avg = sum(vulnerability_levels[current][v] * severity_weights[v] for v in vulnerability_levels[current]) / len(vulnerability_levels[current])
    next_avg = sum(vulnerability_levels[next_][v] * severity_weights[v] for v in vulnerability_levels[next_]) / len(vulnerability_levels[next_])
    improvement = 50 + 30 * (curr_avg - next_avg)
    penalty = switching_costs[current][next_]
    return max(50, improvement - penalty)


# ===============================
# Q-Learning Execution
# ===============================
def q_learning():
    global EPSILON
    ql_rewards, random_rewards = [], []

    for epoch in range(EPOCHS):
        current_service = random.choice(services)
        ql_total, rand_total = 0, 0

        for _ in range(MAX_STEPS_PER_EPOCH):
            update_vulnerabilities()
            hackers_action()

            current_index = services.index(current_service)

            # Q-learning agent
            if random.uniform(0, 1) < EPSILON:
                next_service = random.choice(services)
            else:
                next_index = np.argmax(Q_table[current_index])
                next_service = services[next_index]

            # Random agent
            rand_service = random.choice(services)

            # Rewards
            ql_reward = get_reward(current_service, next_service)
            rand_reward = get_reward(current_service, rand_service)

            ql_total += ql_reward
            rand_total += rand_reward

            # Q-table update
            next_idx = services.index(next_service)
            future_q = np.max(Q_table[next_idx])
            Q_table[current_index][next_idx] = (
                (1 - LEARNING_RATE) * Q_table[current_index][next_idx] +
                LEARNING_RATE * (ql_reward + DISCOUNT_FACTOR * future_q)
            )

            service_frequency[next_service] += 1
            current_service = next_service

        EPSILON = EPSILON_DECAY(epoch)
        ql_rewards.append(ql_total)
        random_rewards.append(rand_total)

    return ql_rewards, random_rewards


# ===============================
# Visualization Functions
# ===============================
def visualize_comparison(ql, rand):
    episodes = range(EPOCHS)
    df_ql = pd.DataFrame({"Episode": episodes, "Reward": ql, "Agent": "QL"})
    df_rand = pd.DataFrame({"Episode": episodes, "Reward": rand, "Agent": "Random"})
    combined = pd.concat([df_ql, df_rand])
    combined["Smoothed"] = combined.groupby("Agent")["Reward"].transform(lambda x: x.rolling(50, min_periods=1).mean())

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=combined, x="Episode", y="Smoothed", hue="Agent", linewidth=2.5)
    plt.title("QL Agent vs Random Agent Reward Trend", fontsize=16)
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("agent_comparison_plot.pdf")
    plt.show()


def visualize_service_frequency():
    total = sum(service_frequency.values())
    percentages = {k: (v / total) * 100 for k, v in service_frequency.items()}
    plt.figure(figsize=(10, 6))
    bars = plt.bar(percentages.keys(), percentages.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.1f}%", ha='center')
    plt.axhline(100 / len(services), color='gray', linestyle='--', label='Uniform Baseline')
    plt.title("Service Switching Frequency")
    plt.ylabel("% of Selections")
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_vulnerability_trends():
    plt.figure(figsize=(12, 6))
    for service, history in vulnerability_history.items():
        smoothed = pd.Series(history).rolling(20).mean()
        plt.plot(smoothed, label=service, linewidth=2)
    plt.title("Vulnerability Trend Over Time")
    plt.ylabel("Avg Vulnerability")
    plt.xlabel("Steps")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("vulnerability_trends.pdf")
    plt.show()


# ===============================
# Run Main
# ===============================
if __name__ == "__main__":
    print("\n[Running Q-Learning Adaptive Defense Simulation]\n")
    ql, rand = q_learning()
    print("\n[Visualizing Agent Reward Comparison]\n")
    visualize_comparison(ql, rand)
    print("\n[Visualizing Service Switching Trends]\n")
    visualize_service_frequency()
    print("\n[Visualizing Vulnerability Evolution]\n")
    visualize_vulnerability_trends()
