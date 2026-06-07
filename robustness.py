"""
robustness.py - Robustness Analysis Module
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class RobustnessAnalyzer:
    """
    Robustness analysis under various stress scenarios

    Scenarios:
    1. Node Failure: 1%, 5%, 10%, 20%
    2. Traffic Burst: 2x, 5x, 10x
    3. Bandwidth Reduction: 10%, 30%, 50%, 70%
    4. Packet Loss: 1%, 5%, 10%
    5. Attack Surge: normal, medium, high, extreme
    """

    def __init__(self, config: dict):
        self.config = config
        self.original_config = config.copy()

    def run_node_failure_test(self, simulation_func, tasks, capacities) -> pd.DataFrame:
        """Test robustness under different node failure probabilities"""

        failure_rates = [0.01, 0.05, 0.10, 0.20]
        results = []

        for rate in failure_rates:
            logger.info(f"Testing node failure rate: {rate * 100}%")

            self.config['network']['edge_failure_prob'] = rate
            result = simulation_func(tasks, capacities, self.config)

            results.append({
                'Scenario': f"Failure {rate * 100}%",
                'Node Failure Rate': rate,
                'Avg Delay (s)': result.get('avg_delay', 0),
                'P95 Delay (s)': result.get('p95_delay', 0),
                'Deadline (%)': result.get('deadline_pct', 0),
                'Fairness': result.get('jain_fairness', 0),
                'Throughput (Mbps)': result.get('throughput', 0)
            })

            self._restore_config()

        return pd.DataFrame(results)

    def run_traffic_burst_test(self, simulation_func, tasks, capacities) -> pd.DataFrame:
        """Test robustness under traffic bursts"""

        burst_factors = [1.0, 2.0, 5.0, 10.0]
        results = []

        for factor in burst_factors:
            logger.info(f"Testing traffic burst factor: {factor}x")

            # Increase arrival rate
            original_rate = self.config['simulation'].get('arrival_rate', 20.0)
            self.config['simulation']['arrival_rate'] = original_rate * factor

            result = simulation_func(tasks, capacities, self.config)

            results.append({
                'Scenario': f"Burst {factor}x",
                'Traffic Factor': factor,
                'Avg Delay (s)': result.get('avg_delay', 0),
                'P95 Delay (s)': result.get('p95_delay', 0),
                'Deadline (%)': result.get('deadline_pct', 0),
                'Fairness': result.get('jain_fairness', 0),
                'Throughput (Mbps)': result.get('throughput', 0)
            })

            self._restore_config()

        return pd.DataFrame(results)

    def run_bandwidth_reduction_test(self, simulation_func, tasks, capacities) -> pd.DataFrame:
        """Test robustness under bandwidth reduction"""

        reduction_factors = [0.1, 0.3, 0.5, 0.7]  # 10%, 30%, 50%, 70% reduction
        results = []

        for factor in reduction_factors:
            logger.info(f"Testing bandwidth reduction: {factor * 100}%")

            bw_min, bw_max = self.config['network']['bandwidth_range']
            self.config['network']['bandwidth_range'] = [bw_min * (1 - factor), bw_max * (1 - factor)]

            result = simulation_func(tasks, capacities, self.config)

            results.append({
                'Scenario': f"BW -{factor * 100}%",
                'Reduction Factor': factor,
                'Avg Delay (s)': result.get('avg_delay', 0),
                'P95 Delay (s)': result.get('p95_delay', 0),
                'Deadline (%)': result.get('deadline_pct', 0),
                'Fairness': result.get('jain_fairness', 0),
                'Throughput (Mbps)': result.get('throughput', 0)
            })

            self._restore_config()

        return pd.DataFrame(results)

    def run_packet_loss_test(self, simulation_func, tasks, capacities) -> pd.DataFrame:
        """Test robustness under packet loss"""

        loss_rates = [0.01, 0.05, 0.10]
        results = []

        for rate in loss_rates:
            logger.info(f"Testing packet loss rate: {rate * 100}%")

            self.config['network']['packet_loss_prob'] = rate
            result = simulation_func(tasks, capacities, self.config)

            results.append({
                'Scenario': f"Loss {rate * 100}%",
                'Loss Rate': rate,
                'Avg Delay (s)': result.get('avg_delay', 0),
                'P95 Delay (s)': result.get('p95_delay', 0),
                'Deadline (%)': result.get('deadline_pct', 0),
                'Fairness': result.get('jain_fairness', 0),
                'Throughput (Mbps)': result.get('throughput', 0)
            })

            self._restore_config()

        return pd.DataFrame(results)

    def run_attack_surge_test(self, simulation_func, tasks, capacities) -> pd.DataFrame:
        """Test robustness under attack surges"""

        attack_levels = {
            'normal': {'attack_ratio': 0.2, 'severity_mult': 1.0},
            'medium': {'attack_ratio': 0.4, 'severity_mult': 1.5},
            'high': {'attack_ratio': 0.6, 'severity_mult': 2.0},
            'extreme': {'attack_ratio': 0.8, 'severity_mult': 3.0}
        }

        results = []

        for level, params in attack_levels.items():
            logger.info(f"Testing attack level: {level}")

            # Modify dataset loading to include more attacks
            self.config['dataset']['attack_ratio'] = params['attack_ratio']

            result = simulation_func(tasks, capacities, self.config)

            results.append({
                'Scenario': f"Attack {level.title()}",
                'Attack Ratio': params['attack_ratio'],
                'Avg Delay (s)': result.get('avg_delay', 0),
                'P95 Delay (s)': result.get('p95_delay', 0),
                'Deadline (%)': result.get('deadline_pct', 0),
                'Fairness': result.get('jain_fairness', 0),
                'Throughput (Mbps)': result.get('throughput', 0)
            })

            self._restore_config()

        return pd.DataFrame(results)

    def run_all_robustness_tests(self, simulation_func, tasks, capacities) -> Dict[str, pd.DataFrame]:
        """Run all robustness tests"""

        return {
            'node_failure': self.run_node_failure_test(simulation_func, tasks, capacities),
            'traffic_burst': self.run_traffic_burst_test(simulation_func, tasks, capacities),
            'bandwidth_reduction': self.run_bandwidth_reduction_test(simulation_func, tasks, capacities),
            'packet_loss': self.run_packet_loss_test(simulation_func, tasks, capacities),
            'attack_surge': self.run_attack_surge_test(simulation_func, tasks, capacities)
        }

    def _restore_config(self):
        """Restore original configuration"""
        self.config['network']['edge_failure_prob'] = 0.01
        self.config['network']['bandwidth_range'] = [5e6, 20e6]
        self.config['network']['packet_loss_prob'] = 0.005
        self.config['simulation']['arrival_rate'] = 20.0
        if 'dataset' in self.config and 'attack_ratio' in self.config['dataset']:
            del self.config['dataset']['attack_ratio']