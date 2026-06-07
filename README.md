# Game-Theoretic Congestion-Aware Task Offloading for Edge-Enabled Industrial IoT Networks

This repository contains the complete implementation, experimental results, and supplementary materials for the paper:

**"Game-Theoretic Congestion-Aware Task Offloading for Edge-Enabled Industrial IoT Networks"**

The proposed framework formulates task offloading in Industrial Internet of Things (IIoT) environments as an exact potential congestion game, enabling decentralized task allocation with provable Nash equilibrium convergence, low latency, and high fairness without requiring reinforcement learning training.

---

## Abstract

Industrial Internet of Things (IIoT) applications generate large volumes of latency-sensitive tasks that must be processed efficiently at edge nodes. Traditional heuristic and reinforcement learning approaches often suffer from congestion unawareness, training overhead, or lack of theoretical guarantees.

This work proposes a game-theoretic congestion-aware task offloading framework that:

- Models task offloading as a non-cooperative congestion game.
- Provides an exact potential function formulation.
- Guarantees pure-strategy Nash equilibrium existence.
- Ensures finite-step convergence of best-response dynamics.
- Achieves low computational complexity.
- Outperforms heuristic and deep reinforcement learning baselines.

Experimental evaluation is conducted on the Edge-IIoTset benchmark using extensive Monte Carlo simulations.

---

## Repository Structure

```text
.
├── figures/                    # Generated figures used in the paper
├── results/                    # Simulation outputs and experimental results
├── tables/                     # Tables reported in the manuscript
│
├── main.py                     # Main execution script
├── config.yaml                 # Experiment configuration
│
├── dataset_loader.py           # Edge-IIoTset dataset processing
├── network_model.py            # Network and communication model
├── queue_model.py              # Queueing and delay model
├── game_model.py               # Exact potential game formulation
├── offloading_algorithm.py     # Nash equilibrium offloading algorithm
│
├── baselines.py                # Baseline methods (Cloud, RR, Greedy, RL)
├── evaluation.py              # Performance evaluation
├── statistical_analysis.py    # Confidence intervals and significance tests
├── robustness.py              # Robustness experiments
├── ablation.py                # Ablation studies
├── visualization.py           # Plot generation
│
├── requirements.txt           # Python dependencies
├── README.md
└── .gitignore
```

---

## Methodology

The proposed framework consists of:

### 1. Delay Modeling

Total delay is decomposed into:

- Transmission delay
- Processing delay
- Congestion delay

allowing congestion effects to be explicitly incorporated into task utilities.

### 2. Exact Potential Game

Task offloading decisions are modeled as player actions within an exact potential game.

The framework provides:

- Potential function existence
- Nash equilibrium existence
- Finite convergence guarantees

### 3. Congestion-Aware Offloading

Tasks iteratively update their selected edge node through best-response dynamics until equilibrium is reached.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/tezatpour-arch/Game-Theoretic-Congestion-Aware-Task-Offloading-for-Edge-Enabled-Industrial-IoT-Networks.git

cd Game-Theoretic-Congestion-Aware-Task-Offloading-for-Edge-Enabled-Industrial-IoT-Networks
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Experiments

Run the complete evaluation:

```bash
python main.py
```

Generate figures:

```bash
python visualization.py
```

Run robustness analysis:

```bash
python robustness.py
```

Run ablation studies:

```bash
python ablation.py
```

Perform statistical analysis:

```bash
python statistical_analysis.py
```

---

## Experimental Results

The proposed method achieved:

| Metric | Value |
|----------|----------|
| Average Delay | 0.413 s |
| Deadline Satisfaction | 94.2% |
| Jain Fairness Index | 0.946 |
| Throughput | 5.85 Mbps |
| Average Convergence Iterations | 2.97 |

Compared with the strongest reinforcement learning baseline (MARL), the proposed framework provides:

- Lower latency
- Higher deadline satisfaction
- Better fairness
- No training overhead
- Provable convergence guarantees

---

## Dataset

Experiments are based on:

**Edge-IIoTset**

Ferrag et al., IEEE Access, 2022.

The dataset contains realistic Industrial IoT traffic and cyber-attack scenarios suitable for evaluating edge computing systems.

---

## Reproducibility

The repository includes:

- Source code
- Configuration files
- Experimental outputs
- Statistical evaluation scripts
- Figure generation scripts

allowing complete reproduction of all reported results.

---

## Citation

If you use this repository, please cite:

```bibtex
@article{tezatpour2025game,
  title={Game-Theoretic Congestion-Aware Task Offloading for Edge-Enabled Industrial IoT Networks},
  author={Tezatpour, Mohammad},
  year={2025}
}
```

---

## License

This project is released for academic and research purposes.

---

## Contact

Mohammad Tezatpour

GitHub:
https://github.com/tezatpour-arch

Repository:
https://github.com/tezatpour-arch/Game-Theoretic-Congestion-Aware-Task-Offloading-for-Edge-Enabled-Industrial-IoT-Networks
