"""
Dataset Loader for Edge-IIoTset
Loads and maps cybersecurity dataset to IIoT tasks
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Represents an IIoT computational task"""
    task_id: int
    priority: float  # P_i in (0, 1]
    data_size: float  # bits
    deadline: float  # absolute time (seconds)
    arrival_time: float  # seconds
    is_critical: bool
    attack_type: str
    packet_size: float
    flow_duration: float
    protocol: str


class EdgeIIoTDatasetLoader:
    """
    Loads Edge-IIoTset dataset and maps to IIoT tasks.

    Mapping Strategy:
    1. Packet Size → Task Size (bits)
    2. Attack Label → Priority & Criticality
    3. Flow Duration → Processing Requirement
    4. Protocol → Device Type
    """

    def __init__(self, config: dict):
        self.config = config
        self.severity_map = {
            'BENIGN': 0.2,
            'DDOS': 0.95,
            'DOS': 0.90,
            'MITM': 0.85,
            'INJECTION': 0.80,
            'MALWARE': 0.85,
            'SCANNING': 0.60,
            'BRUTEFORCE': 0.70
        }

    def load_dataset(self, path: str, max_tasks: int = None) -> pd.DataFrame:
        """Load CSV dataset"""
        logger.info(f"Loading dataset from {path}")

        try:
            df = pd.read_csv(path, low_memory=False, nrows=50000)
            logger.info(f"Loaded {len(df)} records")

            if max_tasks and len(df) > max_tasks:
                df = df.sample(n=max_tasks, random_state=self.config['simulation']['random_seed'])
                logger.info(f"Sampled to {len(df)} tasks")

            return df
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return self._generate_synthetic_dataset(max_tasks or 1000)

    def _get_severity_score(self, attack_type: str) -> float:
        """Map attack type to severity score"""
        for key, score in self.severity_map.items():
            if key in str(attack_type).upper():
                return score
        return 0.3

    def _map_to_task(self, row: pd.Series, task_id: int, arrival_time: float) -> Task:
        """Convert a CSV row to a Task object"""

        # Extract features
        attack_type = str(row.get('Attack_type', 'BENIGN'))
        packet_size = float(row.get('Packet_Length', 500)) if pd.notna(row.get('Packet_Length')) else 500
        flow_duration = float(row.get('Flow_Duration', 100)) if pd.notna(row.get('Flow_Duration')) else 100

        # Determine criticality and priority
        severity = self._get_severity_score(attack_type)
        is_critical = severity > 0.6

        if is_critical:
            priority = np.random.uniform(0.7, 1.0)
            deadline = arrival_time + np.random.uniform(1.0, 2.5)
        else:
            priority = np.random.uniform(0.2, 0.6)
            deadline = arrival_time + np.random.uniform(2.5, 6.0)

        # Task size based on packet size (bits)
        data_size = packet_size * 8 * (1 + flow_duration / 1000)

        return Task(
            task_id=task_id,
            priority=priority,
            data_size=data_size,
            deadline=deadline,
            arrival_time=arrival_time,
            is_critical=is_critical,
            attack_type=attack_type,
            packet_size=packet_size,
            flow_duration=flow_duration,
            protocol=str(row.get('Protocol', 'TCP'))
        )

    def _generate_synthetic_dataset(self, num_tasks: int) -> pd.DataFrame:
        """Fallback: generate synthetic dataset"""
        logger.warning(f"Generating synthetic dataset with {num_tasks} tasks")

        data = []
        for i in range(num_tasks):
            is_attack = np.random.random() < 0.4
            row = {
                'Attack_type': 'DDOS' if is_attack else 'BENIGN',
                'Packet_Length': np.random.uniform(64, 1500),
                'Flow_Duration': np.random.uniform(10, 500),
                'Protocol': np.random.choice(['TCP', 'UDP', 'HTTP', 'MQTT'])
            }
            data.append(row)

        return pd.DataFrame(data)

    def generate_task_stream(self, df: pd.DataFrame, arrival_rate: float = 20.0) -> List[Task]:
        """
        Generate time-ordered task stream from DataFrame

        Args:
            df: Loaded DataFrame
            arrival_rate: Tasks per second (Poisson process)
        """
        tasks = []
        current_time = 0.0

        # Generate inter-arrival times using Poisson process
        inter_arrivals = np.random.exponential(1.0 / arrival_rate, len(df))

        for idx, (_, row) in enumerate(df.iterrows()):
            current_time += inter_arrivals[idx]
            task = self._map_to_task(row, idx, current_time)
            tasks.append(task)

        logger.info(f"Generated {len(tasks)} tasks with arrival rate {arrival_rate} tasks/s")
        return tasks


if __name__ == "__main__":
    # Test dataset loader
    config = {'simulation': {'random_seed': 42}}
    loader = EdgeIIoTDatasetLoader(config)
    df = loader.load_dataset("Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv",
                             max_tasks=500)
    tasks = loader.generate_task_stream(df, arrival_rate=15)
    print(f"Sample task: {tasks[0]}")