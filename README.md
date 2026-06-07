# Game-Theoretic Congestion-Aware Task Offloading for Edge-Enabled IIoT Networks

> **Paper:** "Game-Theoretic Congestion-Aware Task Offloading for Edge-Enabled Industrial IoT Networks"  
> **Authors:** Taha Ezatpour, Arash Anoushiravani, Mohammad Reza Hosseinnejad, Dr. Ali Bozorgmehr  
> **Affiliation:** Islamic Revolution Comprehensive University, Iran

---

## Overview

This repository provides the complete simulation code for the paper. We model IIoT task offloading as a **non-cooperative congestion game** and prove:

- Existence of a **pure-strategy Nash Equilibrium** via an exact potential function (Proposition 2)
- **Finite-step convergence** of best-response dynamics (Theorem 1)

The proposed algorithm is compared against five baselines on a synthetic dataset mirroring CICIDS2017 statistics.

---

## Repository Structure

```
iiot-game-offloading/
│
├── simulation.py          ← Main entry point (run this)
│
├── figures/
│   ├── fig_comparison.pdf   ← Figure 1: 6-panel comparison
│   ├── fig_convergence.pdf  ← Figure 2: Nash equilibrium convergence
│   └── fig_sensitivity.pdf  ← Figure 3: Sensitivity to α, β, γ
│
└── results/
    └── sensitivity_analysis.csv
```

---

## Requirements

```bash
pip install numpy scipy pandas matplotlib seaborn simpy
```

Python ≥ 3.9 recommended.

---

## Reproducing All Results

```bash
git clone https://github.com/<your-username>/iiot-game-offloading.git
cd iiot-game-offloading
pip install -r requirements.txt
python simulation.py
```

This runs:
1. Synthetic CICIDS2017-like dataset generation (5 000 tasks)
2. Monte Carlo comparison (30 runs × 6 policies)
3. All figures saved to `figures/`
4. Sensitivity analysis over α, β, γ

Expected runtime: ~5 minutes on a standard laptop.

---

## Key Results (Small Scale: 3 nodes, 5 000 tasks)

| Policy | Avg Delay (s) | Deadline % | Jain Fairness |
|---|---|---|---|
| Cloud-Only | 170.53 | 3.4 | 0.000 |
| Static Priority | 170.53 | 3.4 | 0.000 |
| Greedy | 18.54 | 91.7 | 1.000 |
| Fuzzy Adaptive | 17.02 | 91.6 | 1.000 |
| Knapsack | 16.88 | 91.6 | 1.000 |
| **Game-Theoretic (Ours)** | **4.63** | **100.0** | **1.000** |

---

## Game-Theoretic Model

The utility function (Eq. 6 in paper):

$$U_i(s_i, \mathbf{s}_{-i}) = \alpha P_i - \beta \hat{C}_{s_i}(\mathbf{s}) - \gamma D_{i,s_i}$$

The exact potential function (Eq. 8):

$$\Phi(\mathbf{s}) = \sum_{i} \alpha P_i \;-\; \beta \sum_j \sum_{k=1}^{n_j(\mathbf{s})} \frac{k}{K_j} \;-\; \gamma \sum_i D_{i,s_i}$$

Default weights: **α = 0.4, β = 0.3, γ = 0.3**

---

## Sensitivity Analysis

The file `figures/fig_sensitivity.pdf` shows how average delay, deadline satisfaction, and convergence speed respond to changes in α, β, γ — addressing Reviewer concern about parameter sensitivity (Proposition 1).

---

## Dataset

We use a **synthetic dataset** that mirrors CICIDS2017 statistical properties:

| Attack Type | Share | Priority Range | Critical |
|---|---|---|---|
| Normal | 65% | U(0.1, 0.5) | No |
| DDoS | 12% | U(0.7, 1.0) | **Yes** |
| DoS Hulk | 8% | U(0.6, 0.9) | **Yes** |
| DoS GoldenEye | 5% | U(0.6, 0.9) | **Yes** |
| PortScan | 6% | U(0.3, 0.6) | No |
| BruteForce | 4% | U(0.2, 0.5) | No |

Arrival process: Poisson(λ = 30 tasks/s) over 300 s window.

To use the **real CICIDS2017 dataset**, download from [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html) and replace `generate_dataset()` with your loader.

---

## Citation

```bibtex
@article{ezatpour2025game,
  title   = {Game-Theoretic Congestion-Aware Task Offloading
             for Edge-Enabled Industrial IoT Networks},
  author  = {Ezatpour, Taha and Anoushiravani, Arash and
             Hosseinnejad, Mohammad Reza and Bozorgmehr, Ali},
  journal = {[Journal Name]},
  year    = {2025}
}
```

---

## License

MIT License — see `LICENSE` for details.
