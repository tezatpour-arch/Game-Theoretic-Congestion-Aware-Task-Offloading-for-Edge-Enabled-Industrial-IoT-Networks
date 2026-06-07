"""
Visualization Module
Generates all plots for the paper: convergence, sensitivity, CDF, boxplots, heatmaps
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import pandas as pd
from matplotlib.ticker import PercentFormatter
import logging

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """
    Generates publication-quality figures for the paper
    """

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = "figures"
        self.dpi = config['output']['dpi']
        self.fig_format = config['output']['figure_format']

        # Set style for publication
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("Set2")

        # Set font sizes
        plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        })

    def plot_convergence(self, history: List[Dict], save_name: str = "convergence"):
        """
        Plot convergence of potential function and number of changes
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Extract data
        iterations = [h['iteration'] for h in history]
        potential = [h['potential'] for h in history]
        changes = [h['changes'] for h in history]

        # Plot potential function
        ax1.plot(iterations, potential, 'b-o', linewidth=2, markersize=4, label='Potential Φ(s)')
        ax1.set_xlabel('Best-Response Iteration')
        ax1.set_ylabel('Potential Function Value')
        ax1.set_title('(a) Potential Function Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot number of changes
        ax2.plot(iterations, changes, 'r-s', linewidth=2, markersize=4, label='Strategy Changes')
        ax2.set_xlabel('Best-Response Iteration')
        ax2.set_ylabel('Number of Strategy Changes')
        ax2.set_title('(b) Best-Response Dynamics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}.{self.fig_format}", dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved convergence plot to {self.output_dir}/{save_name}.{self.fig_format}")

    def plot_sensitivity(self, sensitivity_data: Dict, save_name: str = "sensitivity"):
        """
        Plot sensitivity analysis for α, β, γ, λ parameters
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        params = ['alpha', 'beta', 'gamma', 'lambda_energy']
        titles = [
            r'(a) Sensitivity to $\alpha$ (Priority Weight)',
            r'(b) Sensitivity to $\beta$ (Congestion Weight)',
            r'(c) Sensitivity to $\gamma$ (Delay Weight)',
            r'(d) Sensitivity to $\lambda$ (Energy Weight)'
        ]

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

        for idx, (param, title, color) in enumerate(zip(params, titles, colors)):
            if param in sensitivity_data:
                data = sensitivity_data[param]
                values = data['values']

                # Plot delay
                ax = axes[idx]
                ax_twin = ax.twinx()

                delay_means = [d['mean'] for d in data['delay']]
                delay_stds = [d['std'] for d in data['delay']]

                deadline_means = [d['mean'] for d in data['deadline']]
                deadline_stds = [d['std'] for d in data['deadline']]

                ax.errorbar(values, delay_means, yerr=delay_stds,
                            color=color, marker='o', linewidth=2, markersize=6,
                            capsize=4, label='Average Delay')
                ax.set_xlabel(param)
                ax.set_ylabel('Average Delay (s)', color=color)
                ax.tick_params(axis='y', labelcolor=color)

                ax_twin.errorbar(values, deadline_means, yerr=deadline_stds,
                                 color='orange', marker='s', linewidth=2, markersize=6,
                                 capsize=4, label='Deadline Satisfaction')
                ax_twin.set_ylabel('Deadline Satisfaction (%)', color='orange')
                ax_twin.tick_params(axis='y', labelcolor='orange')

                ax.set_title(title)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}.{self.fig_format}", dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved sensitivity plot to {self.output_dir}/{save_name}.{self.fig_format}")

    def plot_comparison(self, results: Dict, save_name: str = "comparison"):
        """
        Plot comparison of different methods (bar chart with error bars)
        """
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()

        methods = list(results.keys())
        method_names = [m.replace('_', ' ').title() for m in methods]

        # Metrics to plot
        metrics = [
            ('avg_delay', 'Average Delay (s)', 'lower better'),
            ('p95_delay', '95th Percentile Delay (s)', 'lower better'),
            ('deadline_satisfaction', 'Deadline Satisfaction (%)', 'higher better'),
            ('jain_fairness', 'Jain Fairness Index', 'higher better'),
            ('energy_consumption', 'Energy Consumption (J)', 'lower better'),
            ('load_balance', 'Load Balance (Std)', 'lower better')
        ]

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

        for idx, (metric, ylabel, direction) in enumerate(metrics):
            ax = axes[idx]

            means = []
            errors = []

            for method in methods:
                if metric in results[method]:
                    means.append(results[method][metric]['mean'])
                    errors.append(results[method][metric].get('ci_95', results[method][metric]['std']))
                else:
                    means.append(0)
                    errors.append(0)

            bars = ax.bar(method_names, means, yerr=errors, capsize=5,
                          color=colors[idx % len(colors)], edgecolor='black', linewidth=0.8)

            # Highlight the proposed method (assuming 'game_theoretic' is last or identified)
            if 'game_theoretic' in methods:
                gt_idx = methods.index('game_theoretic')
                bars[gt_idx].set_color('#e74c3c')
                bars[gt_idx].set_edgecolor('darkred')
                bars[gt_idx].set_linewidth(2)

            ax.set_ylabel(ylabel)
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            ax.set_title(ylabel)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}.{self.fig_format}", dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved comparison plot to {self.output_dir}/{save_name}.{self.fig_format}")

    def plot_cdf(self, delays_dict: Dict[str, np.ndarray], save_name: str = "cdf_delays"):
        """
        Plot Cumulative Distribution Function of delays
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

        for idx, (method, delays) in enumerate(delays_dict.items()):
            if len(delays) > 0:
                sorted_delays = np.sort(delays)
                cdf = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)
                ax.plot(sorted_delays, cdf, linewidth=2,
                        label=method.replace('_', ' ').title(),
                        color=colors[idx % len(colors)])

        ax.set_xlabel('Delay (seconds)')
        ax.set_ylabel('CDF')
        ax.set_title('Delay Distribution Comparison')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}.{self.fig_format}", dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved CDF plot to {self.output_dir}/{save_name}.{self.fig_format}")

    def plot_boxplots(self, data_dict: Dict[str, List[float]],
                      metric_name: str = "delay", save_name: str = "boxplots"):
        """
        Plot boxplots for comparing distributions
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        data = list(data_dict.values())
        labels = [k.replace('_', ' ').title() for k in data_dict.keys()]

        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True)

        # Color boxes
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel(f'{metric_name.replace("_", " ").title()} (s)')
        ax.set_title(f'Distribution of {metric_name.replace("_", " ").title()}')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}.{self.fig_format}", dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved boxplot to {self.output_dir}/{save_name}.{self.fig_format}")

    def plot_scalability(self, scalability_data: Dict, save_name: str = "scalability"):
        """
        Plot scalability analysis (delay vs number of tasks/nodes)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Delay vs number of tasks
        tasks_sizes = list(scalability_data['tasks'].keys())
        delays = [scalability_data['tasks'][n]['avg_delay'] for n in tasks_sizes]
        errors = [scalability_data['tasks'][n]['delay_std'] for n in tasks_sizes]

        ax1.errorbar(tasks_sizes, delays, yerr=errors,
                     marker='o', linewidth=2, markersize=8, capsize=5,
                     color='#3498db')
        ax1.set_xlabel('Number of Tasks')
        ax1.set_ylabel('Average Delay (s)')
        ax1.set_title('(a) Delay vs Number of Tasks')
        ax1.grid(True, alpha=0.3)

        # Delay vs number of edge nodes
        node_counts = list(scalability_data['nodes'].keys())
        delays_nodes = [scalability_data['nodes'][n]['avg_delay'] for n in node_counts]
        errors_nodes = [scalability_data['nodes'][n]['delay_std'] for n in node_counts]

        ax2.errorbar(node_counts, delays_nodes, yerr=errors_nodes,
                     marker='s', linewidth=2, markersize=8, capsize=5,
                     color='#e74c3c')
        ax2.set_xlabel('Number of Edge Nodes')
        ax2.set_ylabel('Average Delay (s)')
        ax2.set_title('(b) Delay vs Number of Edge Nodes')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}.{self.fig_format}", dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved scalability plot to {self.output_dir}/{save_name}.{self.fig_format}")

    def plot_heatmap(self, matrix: np.ndarray, x_labels: List[str], y_labels: List[str],
                     title: str, save_name: str = "heatmap"):
        """
        Plot heatmap for correlation or load distribution
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

        # Show all ticks
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

        # Rotate tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label('Value')

        ax.set_title(title)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}.{self.fig_format}", dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved heatmap to {self.output_dir}/{save_name}.{self.fig_format}")

    def plot_fairness(self, loads_dict: Dict[str, np.ndarray], capacities: np.ndarray,
                      save_name: str = "fairness"):
        """
        Plot load distribution and fairness comparison
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Normalized load distribution
        methods = list(loads_dict.keys())
        num_edges = len(capacities)

        # Prepare data for grouped bar chart
        x = np.arange(num_edges)
        width = 0.8 / len(methods)

        for idx, (method, loads) in enumerate(loads_dict.items()):
            normalized = loads / capacities
            offset = (idx - len(methods) / 2) * width
            ax1.bar(x + offset, normalized, width,
                    label=method.replace('_', ' ').title(),
                    alpha=0.7)

        ax1.set_xlabel('Edge Node ID')
        ax1.set_ylabel('Normalized Load')
        ax1.set_title('(a) Load Distribution Across Edge Nodes')
        ax1.set_xticks(x)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Fairness values
        fairness_values = []
        method_names = []

        for method, loads in loads_dict.items():
            normalized = loads / capacities
            if np.sum(normalized) > 0:
                fairness = (np.sum(normalized) ** 2) / (num_edges * np.sum(normalized ** 2))
            else:
                fairness = 0
            fairness_values.append(fairness)
            method_names.append(method.replace('_', ' ').title())

        bars = ax2.bar(method_names, fairness_values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        ax2.set_ylabel('Jain Fairness Index')
        ax2.set_title('(b) Fairness Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')

        # Highlight proposed method
        if 'Game Theoretic' in method_names:
            idx = method_names.index('Game Theoretic')
            bars[idx].set_color('#e74c3c')
            bars[idx].set_edgecolor('darkred')
            bars[idx].set_linewidth(2)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}.{self.fig_format}", dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved fairness plot to {self.output_dir}/{save_name}.{self.fig_format}")

    def plot_utility_distribution(self, utilities_dict: Dict[str, np.ndarray],
                                  save_name: str = "utility"):
        """
        Plot utility distribution comparison
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        data = list(utilities_dict.values())
        labels = [k.replace('_', ' ').title() for k in utilities_dict.keys()]

        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True)

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Utility Value')
        ax.set_title('Utility Distribution Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}.{self.fig_format}", dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved utility plot to {self.output_dir}/{save_name}.{self.fig_format}")