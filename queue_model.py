"""
Queue Models for Edge Nodes
M/M/1, M/G/1, and M/M/c implementations
"""

import simpy
import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class QueueModel:
    """Abstract base class for queue models"""

    def __init__(self, env: simpy.Environment, service_rate: float, num_servers: int = 1):
        self.env = env
        self.service_rate = service_rate
        self.num_servers = num_servers
        self.resource = simpy.Resource(env, capacity=num_servers)
        self.queue_length_history = []
        self.waiting_times = []

    def get_service_time(self) -> float:
        """Get service time - to be implemented by child classes"""
        raise NotImplementedError

    def process(self, task_size: float) -> simpy.Event:
        """Process a task through the queue"""
        arrival_time = self.env.now

        with self.resource.request() as request:
            yield request
            waiting_time = self.env.now - arrival_time
            self.waiting_times.append(waiting_time)

            service_time = self.get_service_time()
            yield self.env.timeout(service_time)

        return service_time

    def get_queue_length(self) -> int:
        """Get current queue length"""
        return len(self.resource.queue)

    def record_queue_length(self):
        """Record queue length for monitoring"""
        self.queue_length_history.append((self.env.now, self.get_queue_length()))


class MM1Queue(QueueModel):
    """M/M/1 queue - Exponential inter-arrival and service times"""

    def __init__(self, env: simpy.Environment, service_rate: float):
        super().__init__(env, service_rate, num_servers=1)

    def get_service_time(self) -> float:
        return np.random.exponential(1.0 / self.service_rate)

    def get_expected_delay(self, current_queue: int) -> float:
        """Calculate expected delay using M/M/1 formula"""
        rho = self.env.now / self.service_rate  # Utilization
        if rho >= 1:
            return float('inf')
        return 1.0 / (self.service_rate * (1 - rho))


class MG1Queue(QueueModel):
    """M/G/1 queue - General service time distribution"""

    def __init__(self, env: simpy.Environment, service_rate: float,
                 service_dist: str = "deterministic"):
        super().__init__(env, service_rate, num_servers=1)
        self.service_dist = service_dist

        # Service time parameters
        self.mean_service = 1.0 / service_rate
        self.variance = self.mean_service ** 2

    def get_service_time(self) -> float:
        if self.service_dist == "deterministic":
            return self.mean_service
        elif self.service_dist == "uniform":
            return np.random.uniform(0.5 * self.mean_service, 1.5 * self.mean_service)
        else:  # exponential
            return np.random.exponential(self.mean_service)

    def get_expected_delay(self, current_queue: int) -> float:
        """Pollaczek-Khinchine formula for M/G/1"""
        rho = self.env.now / self.service_rate
        if rho >= 1:
            return float('inf')

        # Pollaczek-Khinchine mean waiting time
        c_v = self.variance ** 0.5 / self.mean_service  # Coefficient of variation
        w_q = (rho * self.mean_service * (1 + c_v ** 2)) / (2 * (1 - rho))
        return w_q + self.mean_service


class MMcQueue(QueueModel):
    """M/M/c queue - Multiple servers"""

    def __init__(self, env: simpy.Environment, service_rate: float, num_servers: int):
        super().__init__(env, service_rate, num_servers=num_servers)

    def get_service_time(self) -> float:
        return np.random.exponential(1.0 / self.service_rate)

    def get_expected_delay(self, current_queue: int) -> float:
        """Erlang-C formula for M/M/c"""
        rho = self.env.now / (self.service_rate * self.num_servers)
        if rho >= 1:
            return float('inf')

        # Approximation for M/M/c
        return 1.0 / (self.service_rate * self.num_servers * (1 - rho))


class QueueFactory:
    """Factory for creating queue models"""

    @staticmethod
    def create_queue(queue_type: str, env: simpy.Environment,
                     service_rate: float, **kwargs) -> QueueModel:
        """
        Create a queue model based on type

        Args:
            queue_type: "M/M/1", "M/G/1", or "M/M/c"
            env: SimPy environment
            service_rate: Service rate (tasks/second)
            **kwargs: Additional parameters
        """
        if queue_type == "M/M/1":
            return MM1Queue(env, service_rate)
        elif queue_type == "M/G/1":
            service_dist = kwargs.get('service_dist', 'exponential')
            return MG1Queue(env, service_rate, service_dist)
        elif queue_type == "M/M/c":
            num_servers = kwargs.get('num_servers', 2)
            return MMcQueue(env, service_rate, num_servers)
        else:
            raise ValueError(f"Unknown queue type: {queue_type}")


if __name__ == "__main__":
    # Test queue models
    env = simpy.Environment()

    # Test M/M/1
    mm1 = MM1Queue(env, service_rate=10.0)
    print(f"M/M/1 expected delay: {mm1.get_expected_delay(0):.4f}s")

    # Test M/G/1
    mg1 = MG1Queue(env, service_rate=10.0, service_dist="deterministic")
    print(f"M/G/1 expected delay: {mg1.get_expected_delay(0):.4f}s")

    # Test M/M/c
    mmc = MMcQueue(env, service_rate=10.0, num_servers=3)
    print(f"M/M/c expected delay: {mmc.get_expected_delay(0):.4f}s")