"""
Network Model for Edge-Enabled IIoT
Includes dynamic bandwidth, latency, and node failures
"""

import simpy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class EdgeNode:
    """Edge Node with processing capabilities"""
    node_id: int
    capacity: float  # tasks per second
    processing_rate: float  # tasks per second
    queue: simpy.Resource
    current_queue_length: float = 0.0
    energy_consumption: float = 0.0
    is_failed: bool = False
    failure_prob: float = 0.01
    recovery_time: float = 10.0

    def get_processing_time(self, task_data_size: float) -> float:
        """Calculate processing time based on data size"""
        return task_data_size / (self.processing_rate * 1e6) + np.random.exponential(0.01)


@dataclass
class Link:
    """Network link between device and edge/cloud"""
    bandwidth: float  # bits per second
    latency: float  # seconds
    packet_loss_prob: float = 0.005

    def get_transmission_time(self, data_size: float) -> float:
        """Calculate transmission time"""
        return data_size / self.bandwidth

    def get_total_delay(self, data_size: float) -> float:
        """Calculate total link delay including transmission and propagation"""
        return self.get_transmission_time(data_size) + self.latency


class DynamicNetwork:
    """
    Dynamic network with time-varying bandwidth and latency
    Simulates realistic IIoT network conditions
    """

    def __init__(self, env: simpy.Environment, config: dict):
        self.env = env
        self.config = config
        self.bandwidth_range = config['network']['bandwidth_range']
        self.latency_range = config['network']['latency_range']
        self.packet_loss_prob = config['network']['packet_loss_prob']

        # Initialize links
        self.edge_links: Dict[int, Link] = {}
        self.cloud_link = Link(
            bandwidth=float(self.bandwidth_range[0] + self.bandwidth_range[1]) / 2.0,
            latency=np.mean(self.latency_range),
            packet_loss_prob=self.packet_loss_prob
        )

        # Start dynamic updates
        self.env.process(self._dynamic_bandwidth_updater())
        self.env.process(self._dynamic_latency_updater())

    def register_edge_link(self, edge_id: int):
        """Register a link to an edge node"""
        self.edge_links[edge_id] = Link(
            bandwidth=np.random.uniform(*self.bandwidth_range),
            latency=np.random.uniform(*self.latency_range),
            packet_loss_prob=self.packet_loss_prob
        )

    def _dynamic_bandwidth_updater(self):
        """Periodically update bandwidth values"""
        while True:
            yield self.env.timeout(5.0)  # Update every 5 seconds

            # Add random variation to bandwidth
            for link in self.edge_links.values():
                variation = np.random.uniform(-0.3, 0.5)
                new_bw = link.bandwidth * (1 + variation)
                link.bandwidth = np.clip(new_bw, self.bandwidth_range[0], self.bandwidth_range[1])

    def _dynamic_latency_updater(self):
        """Periodically update latency values"""
        while True:
            yield self.env.timeout(3.0)  # Update every 3 seconds

            for link in self.edge_links.values():
                variation = np.random.uniform(-0.2, 0.3)
                new_lat = link.latency * (1 + variation)
                link.latency = np.clip(new_lat, self.latency_range[0], self.latency_range[1])

    def simulate_traffic_burst(self, duration: float = 10.0, intensity: float = 3.0):
        """Simulate a traffic burst"""
        original_bw = {eid: link.bandwidth for eid, link in self.edge_links.items()}

        # Reduce bandwidth during burst
        for link in self.edge_links.values():
            link.bandwidth /= intensity

        yield self.env.timeout(duration)

        # Restore bandwidth
        for eid, link in self.edge_links.items():
            link.bandwidth = original_bw[eid]

    def get_edge_delay(self, edge_id: int, data_size: float) -> float:
        """Calculate total delay for edge transmission"""
        if edge_id not in self.edge_links:
            self.register_edge_link(edge_id)

        link = self.edge_links[edge_id]
        return link.get_total_delay(data_size)

    def get_cloud_delay(self, data_size: float) -> float:
        """Calculate total delay for cloud transmission"""
        return self.cloud_link.get_total_delay(data_size)


class NodeFailureSimulator:
    """Simulates node failures and recovery"""

    def __init__(self, env: simpy.Environment, nodes: List[EdgeNode], config: dict):
        self.env = env
        self.nodes = nodes
        self.failure_prob = config['network']['edge_failure_prob']
        self.recovery_time = config['network']['edge_recovery_time']

        # Start failure simulation
        for node in nodes:
            self.env.process(self._node_lifecycle(node))

    def _node_lifecycle(self, node: EdgeNode):
        """Simulate node failures and recovery"""
        while True:
            # Random failure
            failure_time = np.random.exponential(1.0 / node.failure_prob)
            yield self.env.timeout(failure_time)

            if not node.is_failed:
                node.is_failed = True
                logger.info(f"Node {node.node_id} failed at time {self.env.now:.2f}")

                # Recovery
                yield self.env.timeout(node.recovery_time)
                node.is_failed = False
                logger.info(f"Node {node.node_id} recovered at time {self.env.now:.2f}")


class EdgeCloudNetwork:
    """Complete network model with edge nodes and cloud"""

    def __init__(self, env: simpy.Environment, config: dict):
        self.env = env
        self.config = config

        # Initialize edge nodes
        self.edge_nodes: List[EdgeNode] = []
        num_nodes = config['network']['num_edge_nodes']
        capacities = config['network']['edge_capacities']
        processing_rates = config['network']['edge_processing_rates']

        for i in range(num_nodes):
            node = EdgeNode(
                node_id=i,
                capacity=capacities[i],
                processing_rate=processing_rates[i],
                queue=simpy.Resource(env, capacity=int(capacities[i] * 10))
            )
            self.edge_nodes.append(node)

        # Initialize dynamic network
        self.dynamic_network = DynamicNetwork(env, config)

        # Initialize failure simulator
        self.failure_sim = NodeFailureSimulator(env, self.edge_nodes, config)

        # Cloud node
        self.cloud_processing_rate = 50.0  # tasks per second

    def get_available_edge_nodes(self) -> List[int]:
        """Get list of operational edge nodes"""
        return [n.node_id for n in self.edge_nodes if not n.is_failed]

    def get_edge_node(self, node_id: int) -> Optional[EdgeNode]:
        """Get edge node by ID"""
        if node_id < len(self.edge_nodes):
            return self.edge_nodes[node_id]
        return None


if __name__ == "__main__":
    # Test network model
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    env = simpy.Environment()
    network = EdgeCloudNetwork(env, config)
    print(f"Created network with {len(network.edge_nodes)} edge nodes")
    print(f"Available nodes: {network.get_available_edge_nodes()}")