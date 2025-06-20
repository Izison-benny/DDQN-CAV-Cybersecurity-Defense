# cav_env_v3.py

"""
CAV Cybersecurity Environment v3

A simulation environment for testing reinforcement learning agents in cybersecurity-aware connected autonomous vehicle (CAV) systems.

Features:
- 8-dimensional observation space including intrusion, QoS, and resource metrics
- Modular MultiDiscrete action space for language switching per software module
- Attack simulation based on CVSS risk profiles
- Reward function integrates risk, resource, switching penalties, and proactive defense bonuses
"""


import gym
from gym import spaces
import numpy as np
import random
from collections import deque
from typing import Dict, Tuple, Any

# ========================
# Definitions
# ========================
MODULE_CRITICALITY: Dict[str, float] = {
    'Perception': 1.5,
    'Navigation': 1.0,
    'Diagnostics': 0.8
}

MODULE_TO_VERSIONS = {
    'Perception': ['VNF-C++-v1', 'VNF-C++-v2'],
    'Navigation': ['VNF-Java-v1', 'VNF-Java-v2'],
    'Diagnostics': ['VNF-Python-v1', 'VNF-Python-v2', 'VNF-Java-v2']
}

CVSS_SCORES = {
    'C++-v1': 8.8, 'C++-v2': 7.1,
    'Python-v1': 7.8, 'Python-v2': 5.5,
    'Java-v1': 7.2, 'Java-v2': 9.8
}

RESOURCE_PROFILES = {
    'C++-v1': (4, 8, 32), 'C++-v2': (3, 6, 24),
    'Python-v1': (2, 4, 16), 'Python-v2': (1, 2, 8),
    'Java-v1': (3, 6, 24), 'Java-v2': (2, 4, 16)
}

class CAVExecutionEnv(gym.Env):
    metadata = {'render.modes': ['human', 'system']}

    MAX_SWITCHES = 12
    BASE_INTRUSION_PENALTY = -7.5
    BASE_COMPLEX_PENALTY = -15.0
    RISK_PENALTY_FACTOR = 0.03
    RESOURCE_PENALTY_FACTOR = 0.02
    DEFENSE_REWARD = 2.5
    PROACTIVE_DEFENSE_BONUS = 3.0
    SWITCH_PENALTY = 0.3
    BASE_ATTACK_PROB = 0.003
    MAX_ATTACK_PROB = 0.3
    REWARD_SCALE_FACTOR = 0.2
    SECURITY_BONUS_FACTOR = 0.8
    SWITCH_SECURITY_BONUS = 0.7
    COMPLEX_INTRUSION_PROB = 0.015

    def __init__(self, max_steps: int = 300):
        super(CAVExecutionEnv, self).__init__()
        self.module_names = list(MODULE_TO_VERSIONS.keys())
        self.max_steps = max_steps

        self.action_space = spaces.MultiDiscrete([
            len(MODULE_TO_VERSIONS[m]) for m in self.module_names
        ])
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        self.reset()

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.switches = 0
        self.last_action = [0 for _ in self.module_names]
        self.current_action = [0 for _ in self.module_names]
        self.intrusion_counter = 0
        self.complex_intrusion_counter = 0
        self.attack_attempts = 0
        self.defended_attacks = 0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        return np.array([
            self.current_step / self.max_steps,
            self.switches / self.MAX_SWITCHES,
            self.intrusion_counter / max(1, self.current_step),
            self.complex_intrusion_counter / max(1, self.current_step),
            self.attack_attempts / max(1, self.current_step),
            self.defended_attacks / max(1, self.attack_attempts),
            self._normalized_resource_cost(),
            random.random()
        ], dtype=np.float32)

    def _current_version(self, module: str, idx: int) -> str:
        return MODULE_TO_VERSIONS[module][idx]

    def _avg_cvss_score(self) -> float:
        score = 0.0
        for i, module in enumerate(self.module_names):
            version = self._current_version(module, self.current_action[i])
            score += CVSS_SCORES[version]
        return score / len(self.module_names)

    def _normalized_resource_cost(self) -> float:
        total = 0.0
        for i, module in enumerate(self.module_names):
            version = self._current_version(module, self.current_action[i])
            cpu, ram, storage = RESOURCE_PROFILES[version]
            total += cpu + ram + storage
        return total / 100.0

    def _simulate_attack(self) -> Tuple[int, str]:
        self.attack_attempts += 1
        avg_risk = self._avg_cvss_score()
        prob = min(self.MAX_ATTACK_PROB, self.BASE_ATTACK_PROB + avg_risk**1.5 / 250)

        if random.random() < self.COMPLEX_INTRUSION_PROB:
            self.intrusion_counter += 1
            self.complex_intrusion_counter += 1
            return 1, 'complex'
        elif random.random() < prob:
            self.intrusion_counter += 1
            return 1, 'basic'
        else:
            self.defended_attacks += 1
            return 0, 'defended'

    def _evaluate_security_level(self) -> str:
        rate = self.intrusion_counter / max(1, self.current_step)
        if rate > 0.3:
            return 'LOW'
        elif rate > 0.1:
            return 'NORMAL'
        else:
            return 'HIGH'

    def step(self, action: list) -> Tuple[np.ndarray, float, bool, dict]:
        self.current_step += 1
        done = self.current_step >= self.max_steps

        switches_now = sum([1 for i in range(len(self.module_names)) if action[i] != self.last_action[i]])
        self.switches += switches_now
        self.last_action = action
        self.current_action = action

        attacked, attack_type = self._simulate_attack()
        reward, reward_components = self._calculate_reward(attacked, attack_type, switches_now)

        info = {
            'attack_status': attacked,
            'attack_type': attack_type if attacked else 'defended',
            'switches': self.switches,
            'security_level': self._evaluate_security_level(),
            'qos': 1.0 if attack_type == 'defended' else 0.85,
            'reward_components': reward_components
        }

        return self._get_obs(), reward, done, info

    def _calculate_reward(self, attacked: int, attack_type: str, switches_now: int) -> Tuple[float, dict]:
        base_reward = 1.0
        avg_risk = self._avg_cvss_score()
        risk_penalty = avg_risk * self.RISK_PENALTY_FACTOR
        resource_penalty = self._normalized_resource_cost() * self.RESOURCE_PENALTY_FACTOR
        switch_penalty = switches_now * self.SWITCH_PENALTY

        security_bonus = 0.0
        current_security = 1 - (self.intrusion_counter / max(1, self.current_step))

        if current_security > 0.85:
            security_bonus = self.SECURITY_BONUS_FACTOR * (current_security - 0.85)

        if switches_now > 0:
            prev_config_risk = self._prev_avg_risk()
            current_risk = self._avg_cvss_score()
            if current_risk < prev_config_risk:
                security_bonus += self.SWITCH_SECURITY_BONUS * (prev_config_risk - current_risk)

        if attacked:
            if attack_type == 'complex':
                attack_penalty = self.BASE_INTRUSION_PENALTY + self.BASE_COMPLEX_PENALTY
            else:
                attack_penalty = self.BASE_INTRUSION_PENALTY
            proactive_bonus = 0.0
        else:
            attack_penalty = 0.0
            proactive_bonus = self.PROACTIVE_DEFENSE_BONUS * (1.0 - avg_risk / 10)

        total_reward = base_reward - risk_penalty - resource_penalty - switch_penalty + attack_penalty + proactive_bonus
        total_reward *= self.REWARD_SCALE_FACTOR

        return total_reward, {
            'base': base_reward,
            'risk_penalty': -risk_penalty,
            'resource_penalty': -resource_penalty,
            'switch_penalty': -switch_penalty,
            'attack_penalty': attack_penalty,
            'proactive_bonus': proactive_bonus,
            'final_reward': total_reward
        }

    def _prev_avg_risk(self) -> float:
        score = 0.0
        for i, module in enumerate(self.module_names):
            version = self._current_version(module, self.last_action[i])
            score += CVSS_SCORES[version]
        return score / len(self.module_names)

    def render(self, mode: str = 'human') -> None:
        if mode == 'human':
            print(f"\\n=== Step {self.current_step}/{self.max_steps} ===")
            print(f"Switches: {self.switches}/{self.MAX_SWITCHES}")
            print(f"Intrusions: {self.intrusion_counter} | Complex: {self.complex_intrusion_counter}")
            print(f"Attack Attempts: {self.attack_attempts} | Defended: {self.defended_attacks}")
            print(f"Security Level: {self._evaluate_security_level()}")
        elif mode == 'system':
            return self._get_obs()

    def close(self):
        pass

if __name__ == "__main__":
    print("CAV Cybersecurity Environment v3.")
'''

# Save the file (save path)
print(" cav_env_v3.py saved successfully!")
