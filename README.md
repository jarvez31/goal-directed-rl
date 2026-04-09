# Goal-Directed Reinforcement Learning in Avian Brain

A computational model of **Location Cells** and **Path Cells** in the avian brain, built on a Velocity-Driven Oscillatory Network (VDON) with a value-function-based foraging network trained via Temporal Difference (TD) learning.

This is the codebase accompanying the thesis:
> *A Unified Hierarchical Value Based Oscillatory Network of Location Cells and Path Cells in Avian Brain*
> Bharat K. Patil — Dual Degree (B.Tech & M.Tech), Dept. of Aerospace Engineering, IIT Madras, 2020.

---

## Overview

Birds achieve remarkable spatial navigation using specialised cells in the avian hippocampal formation — **Location Cells** (high spatial selectivity near rewarding sites) and **Path Cells** (elevated firing along trajectories between reward locations). This project proposes a biologically plausible model that reproduces these cells using only velocity and head direction as inputs.

The model pipeline:
1. **VDON** (Velocity Driven Oscillatory Network) — encodes spatial information from speed and head direction into grid-cell-like and place-cell-like representations via two stacked LAHN layers.
2. **Value Network** — an MLP trained with TD learning over the LAHN 2 output to build a reward-based value space.
3. **GEN (Go-Explore-NoGo)** — a hill-climbing algorithm used at test time to generate goal-directed trajectories based on the learned value gradient.

The model was tested in two environments: a **Plus Maze** (matching the classic homing pigeon experiment by Siegel et al., 2005) and a **Box Environment** with four rewarding bowl regions.

---

## Repository Structure

```
goal-directed-rl/
├── plus_maze/              # Code for the Plus Maze environment
│   ├── first_train.py      # Stage 1: train value model on free trajectory
│   ├── retrain2.py         # Stage 2: further train on temporally rewarded trajectory
│   ├── networks.py         # Actor and Critic network class definitions
│   ├── plot_value.py       # Plotting value space and neuron firing patterns
│   ├── real_vis_traj.py    # Real-time trajectory visualisation
│   ├── GEN.py              # Main script to generate GEN-based trajectories
│   ├── GEN_func.py         # Helper functions for GEN
│   ├── corr_xy.py          # Maps LAHN output to spatial coordinates
│   └── actor_*.py          # Actor network scripts
│
├── box_env/                # Code for the Box Environment (same structure as above)
│   ├── first_train.py
│   ├── retrain2.py
│   ├── networks.py
│   ├── plot_value.py
│   ├── real_vis_traj.py
│   ├── GEN.py
│   ├── GEN_func.py
│   ├── corr_xy.py
│   └── actor_*.py
│
└── DDP_Thesis_AE15B046.pdf # Full thesis
```

> **Note:** Both folders share the same file names but contain environment-specific implementations.

---

## Key Components

### `first_train.py`
Trains the value network (MLP) on the first trajectory — a free random walk where reward is given continuously inside bowl regions. Uses TD learning to build an initial value space.

### `retrain2.py`
Fine-tunes the value network on the second trajectory, where reward is presented only for a fixed time window `T` after the agent enters a bowl (mimicking finite food pellets). This is the main training stage for goal-specific behaviour.

### `networks.py`
Contains PyTorch/NumPy class definitions for:
- **Critic network** — outputs value `V(S)` given state `S(t)`
- **Actor network** — outputs action/direction given state `S(t)`

### `GEN.py` / `GEN_func.py`
Implements the **Go-Explore-NoGo** algorithm for test-time trajectory generation. The agent moves by following the value gradient — repeating successful steps (Go), exploring randomly at plateaus (Explore), or reversing when value drops (NoGo).

### `corr_xy.py`
Regression utility that maps LAHN 2 neural responses back to `(x, y)` position coordinates — used to verify that spatial information is preserved in the network's internal representation.

### `plot_value.py` / `real_vis_traj.py`
Plotting utilities for value heat maps, hidden neuron firing patterns, and live trajectory visualisation.

---

## Results Summary

- The value network successfully learned a **reward-correlated spatial map** in both environments.
- **70–75% of hidden layer neurons** showed goal-specific firing near rewarding locations, consistent with Location Cell behaviour observed in homing pigeons.
- The GEN algorithm did not produce a consistent clockwise/anticlockwise visiting pattern — suggesting the need for a dedicated **Actor network** for directed navigation.

---

## What's Not Included

Due to file size constraints (~1 GB of data) and licensing, the following are **not** included in this repository:

| Missing Component | Reason |
|---|---|
| Training data / trajectory datasets | ~1 GB, too large for GitHub |
| VDON model code | Not authored by this project |
| Trajectory generation scripts | Not authored by this project |

To reproduce results, you will need to generate VDON outputs (LAHN 2 activations + trajectory data) separately and place them in the appropriate directories.

---

## References

- Siegel et al. (2005). Spatial-specificity of single-units in the hippocampal formation of homing pigeons. *Hippocampus*.
- Hough & Bingman (2004). Spatial response properties of homing pigeon hippocampal neurons. *J. Comparative Physiology A*.
- Chakravarthy & Moustafa (2018). *Computational Neuroscience Models of the Basal Ganglia*. Springer.
- Hafting et al. (2005). Microstructure of a spatial map in the entorhinal cortex. *Nature*.

---

## Citation

If you use this work, please cite:

```
Bharat K. Patil, "A Unified Hierarchical Value Based Oscillatory Network of Location Cells
and Path Cells in Avian Brain", Dual Degree Thesis, IIT Madras, June 2020.
```
