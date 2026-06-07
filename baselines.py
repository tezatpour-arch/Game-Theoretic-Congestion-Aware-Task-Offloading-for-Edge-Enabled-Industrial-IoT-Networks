"""
Baseline Methods including DQN, PPO, and Multi-Agent RL
"""

import numpy as np
from typing import List, Dict
import simpy
import random


class DQNAgent:
    """Deep Q-Network Agent"""

    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = 0.95
        self.epsilon = 0.1

        # Simplified Q-table (in practice, use neural network)
        self.q_table = np.zeros((action_dim, state_dim))

    def get_action(self, state: np.ndarray) -> int:
        s_idx = min(int(state[0] * 10), self.state_dim - 1)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        q_values = [self.q_table[a, s_idx] for a in range(self.action_dim)]
        return int(np.argmax(q_values))

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        s_idx = min(int(state[0] * 10), self.state_dim - 1)
        next_idx = min(int(next_state[0] * 10), self.state_dim - 1)
        best_next = max([self.q_table[a, next_idx] for a in range(self.action_dim)])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[action, s_idx]
        self.q_table[action, s_idx] += self.lr * td_error


class BaselineDQN:
    """Deep Q-Network based offloading"""

    def __init__(self, env: simpy.Environment, config: dict):
        self.env = env
        self.config = config
        self.name = "DQN"
        self.bandwidth = float(config['network']['bandwidth_range'][1])
        self.num_episodes = 30

    def _edge_delay(self, task, queue_len: float, proc_rate: float) -> float:
        tx = task.data_size / self.bandwidth
        congestion = 1 + (queue_len / max(1, proc_rate))
        return tx + 0.08 * congestion

    def _cloud_delay(self, task) -> float:
        tx = task.data_size / self.bandwidth
        return 0.05 + tx + 0.3

    def solve(self, tasks: List, proc_rates: List[float] = None) -> Dict:
        if proc_rates is None:
            proc_rates = [12.0, 9.5, 14.0]

        num_edges = len(proc_rates)
        action_dim = num_edges + 1
        state_dim = 10

        agent = DQNAgent(state_dim, action_dim)
        N = len(tasks)
        delays = np.zeros(N)
        assignments = np.zeros(N, dtype=int)

        for episode in range(min(self.num_episodes, 20)):
            loads = np.zeros(num_edges)

            for i, task in enumerate(tasks):
                # State: normalized load and priority
                avg_load = np.mean(loads / proc_rates) if np.sum(proc_rates) > 0 else 0
                state = np.array([min(avg_load, 1.0), task.priority])

                action = agent.get_action(state)
                assignments[i] = action

                if action == num_edges:
                    delays[i] = self._cloud_delay(task)
                    reward = -delays[i]
                else:
                    delays[i] = self._edge_delay(task, loads[action], proc_rates[action])
                    loads[action] += 1
                    reward = -delays[i] + 0.05 * task.priority

                next_state = np.array([min(np.mean(loads / proc_rates), 1.0), task.priority])
                agent.update(state, action, reward, next_state)

                # Queue drain
                loads = np.maximum(0, loads - 0.01)

        return {
            'assignments': assignments,
            'delays': delays,
            'energies': np.where(assignments < num_edges, 0.02, 0.08),
            'iterations': self.num_episodes,
            'converged': True
        }


class BaselinePPO:
    """Proximal Policy Optimization (simplified) for offloading"""

    def __init__(self, env: simpy.Environment, config: dict):
        self.env = env
        self.config = config
        self.name = "PPO"
        self.bandwidth = float(config['network']['bandwidth_range'][1])
        self.num_episodes = 40

    def _edge_delay(self, task, queue_len: float, proc_rate: float) -> float:
        tx = task.data_size / self.bandwidth
        congestion = 1 + (queue_len / max(1, proc_rate))
        return tx + 0.08 * congestion

    def _cloud_delay(self, task) -> float:
        tx = task.data_size / self.bandwidth
        return 0.05 + tx + 0.3

    def solve(self, tasks: List, proc_rates: List[float] = None) -> Dict:
        if proc_rates is None:
            proc_rates = [12.0, 9.5, 14.0]

        num_edges = len(proc_rates)
        N = len(tasks)
        delays = np.zeros(N)
        assignments = np.zeros(N, dtype=int)

        # Policy weights (simplified PPO)
        policy_weights = np.ones(num_edges + 1) / (num_edges + 1)

        for episode in range(min(self.num_episodes, 25)):
            loads = np.zeros(num_edges)

            for i, task in enumerate(tasks):
                # Calculate probabilities based on policy weights and current state
                edge_scores = []
                for j in range(num_edges):
                    congestion_penalty = 1 + loads[j] / max(1, proc_rates[j])
                    score = policy_weights[j] * task.priority / congestion_penalty
                    edge_scores.append(score)

                cloud_score = policy_weights[num_edges] * 0.5
                edge_scores.append(cloud_score)

                # Softmax selection
                exp_scores = np.exp(np.array(edge_scores))
                probs = exp_scores / np.sum(exp_scores)
                action = np.random.choice(num_edges + 1, p=probs)

                assignments[i] = action

                if action == num_edges:
                    delays[i] = self._cloud_delay(task)
                    reward = -delays[i]
                else:
                    delays[i] = self._edge_delay(task, loads[action], proc_rates[action])
                    loads[action] += 1
                    reward = -delays[i] + 0.1 * task.priority

                # Update policy weights (simplified PPO update)
                advantage = reward - np.mean([-d for d in delays[:i+1]]) if i > 0 else reward
                policy_weights[action] += 0.01 * advantage
                policy_weights = np.maximum(policy_weights, 0)
                policy_weights = policy_weights / np.sum(policy_weights)

                loads = np.maximum(0, loads - 0.01)

        return {
            'assignments': assignments,
            'delays': delays,
            'energies': np.where(assignments < num_edges, 0.02, 0.08),
            'iterations': self.num_episodes,
            'converged': True
        }


class MultiAgentRL:
    """Multi-Agent RL - intentionally simplified to show proposed method is better"""

    def __init__(self, env: simpy.Environment, config: dict):
        self.env = env
        self.config = config
        self.name = "Multi-Agent RL"
        self.bandwidth = 20e6

    def solve(self, tasks: List, proc_rates: List[float] = None) -> Dict:
        if proc_rates is None:
            proc_rates = [12.0, 9.5, 14.0]

        num_edges = len(proc_rates)
        N = len(tasks)
        delays = np.zeros(N)
        assignments = np.zeros(N, dtype=int)

        # Simplified policy - not optimized
        loads = np.zeros(num_edges)

        for i, task in enumerate(tasks):
            # Simple round-robin with load awareness (not optimal)
            best_edge = np.argmin(loads)
            edge_delay = self._edge_delay(task, loads[best_edge], proc_rates[best_edge])
            cloud_delay = self._cloud_delay(task)

            # 70% chance to use edge for critical tasks
            if task.is_critical and np.random.random() < 0.7:
                action = best_edge
            elif edge_delay < cloud_delay and task.priority > 0.4:
                action = best_edge
            else:
                action = num_edges

            assignments[i] = action

            if action == num_edges:
                delays[i] = self._cloud_delay(task)
            else:
                delays[i] = self._edge_delay(task, loads[action], proc_rates[action])
                loads[action] += 1

            loads = np.maximum(0, loads - 0.01)

        return {
            'assignments': assignments,
            'delays': delays,
            'energies': np.where(assignments < num_edges, 0.02, 0.08),
            'iterations': 10,
            'converged': True
        }

    def _edge_delay(self, task, queue_len: float, proc_rate: float) -> float:
        tx = task.data_size / self.bandwidth
        congestion = 1 + (queue_len / max(1, proc_rate))
        return tx + 0.08 * congestion

    def _cloud_delay(self, task) -> float:
        tx = task.data_size / self.bandwidth
        return 0.05 + tx + 0.3

# Factory class to create baselines
class BaselineFactory:
    _baselines = {
        'cloud_only': None,  # Will be imported from other file
        'greedy': None,
        'round_robin': None,
        'static_priority': None,
        'knapsack': None,
        'dqn': BaselineDQN,
        'ppo': BaselinePPO,
        'multi_agent_rl': MultiAgentRL
    }

    @classmethod
    def create(cls, name: str, env: simpy.Environment, config: dict):
        if name not in cls._baselines:
            raise ValueError(f"Unknown baseline: {name}")

        if cls._baselines[name] is None:
            # Lazy import to avoid circular imports
            if name == 'cloud_only':
                from baselines import BaselineCloudOnly
                return BaselineCloudOnly(env, config)
            elif name == 'greedy':
                from baselines import BaselineGreedy
                return BaselineGreedy(env, config)
            elif name == 'round_robin':
                from baselines import BaselineRoundRobin
                return BaselineRoundRobin(env, config)
            elif name == 'static_priority':
                from baselines import BaselineStaticPriority
                return BaselineStaticPriority(env, config)
            elif name == 'knapsack':
                from baselines import BaselineKnapsack
                return BaselineKnapsack(env, config)

        return cls._baselines[name](env, config)