"""
Game-Theoretic Model for Task Offloading
Non-cooperative congestion game with exact potential function
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlayerState:
    """State of a player (device) in the game"""
    player_id: int
    current_strategy: int  # node ID (0..M-1 for edge, M for cloud)
    utility_history: List[float] = None

    def __post_init__(self):
        self.utility_history = []


class CongestionGame:
    """
    Non-cooperative congestion game for task offloading

    Players: IIoT devices
    Strategies: Edge nodes (0..M-1) or Cloud (M)
    Utility: U_i = α·P_i - β·C_hat - γ·D_i - λ·E_i
    """

    def __init__(self, num_players: int, num_edge_nodes: int, config: dict):
        self.num_players = num_players
        self.num_edge_nodes = num_edge_nodes
        self.num_strategies = num_edge_nodes + 1  # +1 for cloud

        # Game weights
        self.alpha = config['game']['alpha']  # Priority weight
        self.beta = config['game']['beta']  # Congestion weight
        self.gamma = config['game']['gamma']  # Delay weight
        self.lambda_e = config['game']['lambda_energy']  # Energy weight

        # Player states
        self.players: List[PlayerState] = [
            PlayerState(player_id=i, current_strategy=num_edge_nodes)
            for i in range(num_players)
        ]

        # Strategy counts
        self.strategy_counts = np.zeros(self.num_strategies, dtype=int)

        # Potential function history
        self.potential_history = []

        logger.info(f"Initialized CongestionGame with {num_players} players, {self.num_strategies} strategies")

    def get_congestion(self, strategy: int, counts: np.ndarray = None) -> float:
        """Calculate normalized congestion for a strategy"""
        if counts is None:
            counts = self.strategy_counts

        if strategy == self.num_edge_nodes:  # Cloud
            return 0.0
        else:
            # Congestion = queue_length / capacity
            return counts[strategy] / (self.num_edge_nodes * 5.0)  # capacity ~5

    def calculate_utility(self, player_id: int, strategy: int,
                          counts: np.ndarray = None) -> float:
        """
        Calculate utility for a player given a strategy

        U_i = α·P_i - β·C_hat - γ·D_i - λ·E_i
        """
        if counts is None:
            counts = self.strategy_counts

        # Priority term (P_i) - would come from task
        priority = np.random.uniform(0.3, 0.9)  # Placeholder

        # Congestion term
        congestion = self.get_congestion(strategy, counts)

        # Delay term (D_i) - placeholder, will be computed by network
        delay = 0.1 + congestion * 0.5

        # Energy term (E_i)
        energy = 0.05 if strategy == self.num_edge_nodes else 0.02

        # Calculate utility
        utility = (self.alpha * priority -
                   self.beta * congestion -
                   self.gamma * delay -
                   self.lambda_e * energy)

        return utility

    def potential_function(self) -> float:
        """
        Exact potential function for congestion game

        Φ(s) = -β·Σ_j Σ_{k=1}^{n_j} k/K_j - γ·Σ_i D_i(s)
        """
        # Congestion part
        congestion_sum = 0.0
        for j in range(self.num_edge_nodes):
            n = self.strategy_counts[j]
            congestion_sum += n * (n + 1) / (2.0 * (self.num_edge_nodes * 5.0))

        # Potential value
        potential = -self.beta * congestion_sum

        self.potential_history.append(potential)
        return potential

    def best_response(self, player_id: int, current_counts: np.ndarray) -> int:
        """Find best response strategy for a player"""
        best_strategy = self.players[player_id].current_strategy
        best_utility = self.calculate_utility(player_id, best_strategy, current_counts)

        for s in range(self.num_strategies):
            if s == best_strategy:
                continue

            # Tentatively update counts
            if s < self.num_edge_nodes:
                current_counts[s] += 1

            utility = self.calculate_utility(player_id, s, current_counts)

            if utility > best_utility + 1e-6:
                best_utility = utility
                best_strategy = s

            # Revert counts
            if s < self.num_edge_nodes:
                current_counts[s] -= 1

        return best_strategy

    def update_strategy(self, player_id: int, new_strategy: int):
        """Update player's strategy and counts"""
        old_strategy = self.players[player_id].current_strategy

        if old_strategy != new_strategy:
            if old_strategy < self.num_edge_nodes:
                self.strategy_counts[old_strategy] -= 1
            if new_strategy < self.num_edge_nodes:
                self.strategy_counts[new_strategy] += 1

            self.players[player_id].current_strategy = new_strategy

    def is_nash_equilibrium(self) -> bool:
        """Check if current strategy profile is a Nash equilibrium"""
        for player in self.players:
            current_utility = self.calculate_utility(
                player.player_id, player.current_strategy
            )

            for s in range(self.num_strategies):
                if s == player.current_strategy:
                    continue

                utility = self.calculate_utility(player.player_id, s)
                if utility > current_utility + 1e-6:
                    return False

        return True

    def get_equilibrium_gap(self) -> float:
        """Calculate maximum unilateral deviation gain"""
        max_gain = 0.0

        for player in self.players:
            current_utility = self.calculate_utility(
                player.player_id, player.current_strategy
            )

            for s in range(self.num_strategies):
                if s == player.current_strategy:
                    continue

                utility = self.calculate_utility(player.player_id, s)
                gain = utility - current_utility
                max_gain = max(max_gain, gain)

        return max_gain


class NashEquilibriumSolver:
    """
    Solver for Nash equilibrium using best-response dynamics
    """

    def __init__(self, game: CongestionGame, config: dict):
        self.game = game
        self.max_iterations = config['game']['max_iterations']
        self.convergence_threshold = config['game']['convergence_threshold']
        self.convergence_history = []

    def synchronous_best_response(self, priorities: List[float]) -> Dict:
        """
        Synchronous best-response dynamics
        All players update simultaneously
        """
        iteration = 0
        converged = False

        while iteration < self.max_iterations and not converged:
            # Calculate best responses for all players
            new_strategies = []
            counts = self.game.strategy_counts.copy()

            for player in self.game.players:
                best = self.game.best_response(player.player_id, counts)
                new_strategies.append(best)

            # Update all players
            max_change = 0
            for i, player in enumerate(self.game.players):
                if new_strategies[i] != player.current_strategy:
                    max_change += 1
                    self.game.update_strategy(player.player_id, new_strategies[i])

            # Update potential
            potential = self.game.potential_function()
            self.convergence_history.append({
                'iteration': iteration,
                'potential': potential,
                'changes': max_change,
                'is_nash': self.game.is_nash_equilibrium()
            })

            if max_change < self.convergence_threshold:
                converged = True

            iteration += 1

        return {
            'converged': converged,
            'iterations': iteration,
            'final_potential': self.game.potential_function(),
            'strategy_counts': self.game.strategy_counts.copy(),
            'history': self.convergence_history
        }

    def asynchronous_best_response(self, priorities: List[float],
                                   update_prob: float = 0.3) -> Dict:
        """
        Asynchronous best-response dynamics
        Players update with probability update_prob
        """
        iteration = 0
        converged = False

        while iteration < self.max_iterations and not converged:
            changes = 0

            for player in self.game.players:
                if np.random.random() < update_prob:
                    best = self.game.best_response(
                        player.player_id,
                        self.game.strategy_counts.copy()
                    )

                    if best != player.current_strategy:
                        changes += 1
                        self.game.update_strategy(player.player_id, best)

            # Update potential
            potential = self.game.potential_function()
            self.convergence_history.append({
                'iteration': iteration,
                'potential': potential,
                'changes': changes,
                'is_nash': self.game.is_nash_equilibrium()
            })

            if changes == 0:
                converged = True

            iteration += 1

        return {
            'converged': converged,
            'iterations': iteration,
            'final_potential': self.game.potential_function()
        }

    def randomized_best_response(self, priorities: List[float],
                                 random_seed: int = 42) -> Dict:
        """
        Randomized best-response dynamics
        Random order of players each iteration
        """
        np.random.seed(random_seed)
        iteration = 0
        converged = False

        while iteration < self.max_iterations and not converged:
            order = np.random.permutation(len(self.game.players))
            changes = 0

            for player_id in order:
                player = self.game.players[player_id]
                best = self.game.best_response(
                    player.player_id,
                    self.game.strategy_counts.copy()
                )

                if best != player.current_strategy:
                    changes += 1
                    self.game.update_strategy(player.player_id, best)

            potential = self.game.potential_function()
            self.convergence_history.append({
                'iteration': iteration,
                'potential': potential,
                'changes': changes
            })

            if changes == 0:
                converged = True

            iteration += 1

        return {
            'converged': converged,
            'iterations': iteration,
            'final_potential': self.game.potential_function()
        }


if __name__ == "__main__":
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    game = CongestionGame(num_players=100, num_edge_nodes=3, config=config)
    solver = NashEquilibriumSolver(game, config)

    priorities = np.random.uniform(0.1, 1.0, 100)
    result = solver.synchronous_best_response(priorities)

    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Final potential: {result['final_potential']:.4f}")