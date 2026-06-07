"""
ablation.py - Ablation Study Module
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class AblationStudy:
    """
    Ablation study to analyze contribution of each component

    Components:
    1. Full Model (all terms)
    2. No Congestion Term (β=0)
    3. No Energy Term (λ=0)
    4. No Priority Term (α=0)
    5. No Delay Term (γ=0)
    6. No Nash Equilibrium Update (random strategy)
    7. No Best Response (greedy only)
    8. No Dynamic Network (static)
    9. No Failure Handling
    """

    def __init__(self, config: dict):
        self.config = config
        self.original_config = config.copy()

        self.ablation_configs = {
            'Full Model': self._full_model,
            'No Congestion Term': self._no_congestion,
            'No Energy Term': self._no_energy,
            'No Priority Term': self._no_priority,
            'No Delay Term': self._no_delay,
            'No Nash Equilibrium': self._no_nash,
            'No Best Response': self._no_best_response,
            'No Dynamic Network': self._no_dynamic,
            'No Failure Handling': self._no_failure
        }

    def _full_model(self):
        """Full model with all terms"""
        self.config['game']['beta'] = 0.55
        self.config['game']['lambda_energy'] = 0.05
        self.config['game']['alpha'] = 0.25
        self.config['game']['gamma'] = 0.15

    def _no_congestion(self):
        """Remove congestion term (β=0)"""
        self.config['game']['beta'] = 0.0
        self.config['game']['alpha'] = 0.40
        self.config['game']['gamma'] = 0.40
        self.config['game']['lambda_energy'] = 0.20

    def _no_energy(self):
        """Remove energy term (λ=0)"""
        self.config['game']['lambda_energy'] = 0.0
        self.config['game']['alpha'] = 0.28
        self.config['game']['beta'] = 0.58
        self.config['game']['gamma'] = 0.14

    def _no_priority(self):
        """Remove priority term (α=0)"""
        self.config['game']['alpha'] = 0.0
        self.config['game']['beta'] = 0.65
        self.config['game']['gamma'] = 0.25
        self.config['game']['lambda_energy'] = 0.10

    def _no_delay(self):
        """Remove delay term (γ=0)"""
        self.config['game']['gamma'] = 0.0
        self.config['game']['alpha'] = 0.30
        self.config['game']['beta'] = 0.60
        self.config['game']['lambda_energy'] = 0.10

    def _no_nash(self):
        """No Nash equilibrium (random strategy selection)"""
        self.config['game']['max_iterations'] = 0
        self.config['game']['beta'] = 0.0

    def _no_best_response(self):
        """No best-response dynamics (greedy only)"""
        self.config['game']['max_iterations'] = 1

    def _no_dynamic(self):
        """No dynamic network (static bandwidth and latency)"""
        self.config['network']['bandwidth_range'] = [10e6, 10e6]
        self.config['network']['latency_range'] = [0.02, 0.02]

    def _no_failure(self):
        """No failure handling"""
        self.config['network']['edge_failure_prob'] = 0.0

    def run_ablation(self, simulation_func, tasks, capacities) -> pd.DataFrame:
        """Run all ablation configurations"""

        results = []

        for name, config_func in self.ablation_configs.items():
            logger.info(f"Running ablation: {name}")

            # Apply configuration
            config_func()

            # Run simulation (single run, not Monte Carlo for speed)
            result = simulation_func(tasks, capacities, self.config)

            results.append({
                'Configuration': name,
                'Avg Delay (s)': result.get('avg_delay', 0),
                'P95 Delay (s)': result.get('p95_delay', 0),
                'Deadline (%)': result.get('deadline_pct', 0),
                'Fairness': result.get('jain_fairness', 0),
                'Energy (J)': result.get('energy_consumption', 0),
                'Throughput (Mbps)': result.get('throughput', 0),
                'Convergence': result.get('convergence_iterations', 0)
            })

            # Restore original config
            self._restore_config()

        return pd.DataFrame(results)

    def _restore_config(self):
        """Restore original configuration"""
        self.config['game']['alpha'] = 0.25
        self.config['game']['beta'] = 0.55
        self.config['game']['gamma'] = 0.15
        self.config['game']['lambda_energy'] = 0.05
        self.config['game']['max_iterations'] = 30
        self.config['network']['bandwidth_range'] = [5e6, 20e6]
        self.config['network']['latency_range'] = [0.01, 0.05]
        self.config['network']['edge_failure_prob'] = 0.01

    def generate_latex_table(self, df: pd.DataFrame) -> str:
        """Generate LaTeX table for paper"""

        latex = r"\begin{table}[htb]" + "\n"
        latex += r"\centering" + "\n"
        latex += r"\caption{Ablation Study Results}" + "\n"
        latex += r"\label{tab:ablation}" + "\n"
        latex += r"\begin{tabular}{lccccc}" + "\n"
        latex += r"\toprule" + "\n"
        latex += r"\textbf{Configuration} & \textbf{Delay (s)} & \textbf{P95 (s)} & \textbf{Deadline \%} & \textbf{Fairness} & \textbf{Energy} \\" + "\n"
        latex += r"\midrule" + "\n"

        for _, row in df.iterrows():
            latex += f"{row['Configuration']} & "
            latex += f"{row['Avg Delay (s)']:.3f} & "
            latex += f"{row['P95 Delay (s)']:.3f} & "
            latex += f"{row['Deadline (%)']:.1f} & "
            latex += f"{row['Fairness']:.3f} & "
            latex += f"{row['Energy (J)']:.3f} \\\\\n"

        latex += r"\bottomrule" + "\n"
        latex += r"\end{tabular}" + "\n"
        latex += r"\end{table}" + "\n"

        return latex