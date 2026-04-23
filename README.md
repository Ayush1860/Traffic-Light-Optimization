<h1 align="center"> 🚦 Traffic Light Optimization With Deep Reinforcement Learning </h1>

<p align="center">
  <b>A Deep Q-Learning agent that learns to coordinate traffic signals across a 3-intersection corridor using a Transformer-based neural network, Max Pressure control theory, and Fuzzy Logic congestion evaluation.</b>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Complete Pipeline](#complete-pipeline)
  - [Stage 1 — Network Generation](#stage-1--network-generation)
  - [Stage 2 — Traffic Generation](#stage-2--traffic-generation)
  - [Stage 3 — State Observation](#stage-3--state-observation)
  - [Stage 4 — Action Selection](#stage-4--action-selection)
  - [Stage 5 — Reward Computation](#stage-5--reward-computation)
  - [Stage 6 — Learning](#stage-6--learning)
  - [Stage 7 — Testing & Evaluation](#stage-7--testing--evaluation)
- [Model Architecture](#model-architecture)
- [Why Deep RL Over Traditional ML](#why-deep-rl-over-traditional-ml)
- [Training Results](#training-results)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [References](#references)

---

## Overview

This project implements an intelligent Traffic Light Control System (TLCS) that uses **Deep Reinforcement Learning** to optimize traffic flow across a **linear corridor of 3 interconnected traffic-light intersections**. A single RL agent jointly controls all three lights, learning coordinated signal timing to minimize congestion across the entire network.

### Key Features

- **Multi-Intersection Joint Control** — One agent simultaneously manages TL1, TL2, and TL3 through a compact 8-action encoding
- **Transformer-Based DQN** — Self-attention mechanisms allow the model to learn spatial relationships between lanes across intersections
- **Hybrid Reward Function** — Combines wait-time delta reward with Fuzzy Logic congestion scoring
- **Realistic SUMO Simulation** — 4-lane roads, 600 vehicles per episode, Weibull-distributed traffic generation
- **YOLO Perception Module** — YOLOv11-based vehicle detection for real-world deployment readiness

---

## Complete Pipeline

Below is the end-to-end pipeline that makes this system work — from building the road network all the way to evaluating a trained agent.

### Pipeline Overview

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Stage 1    │    │   Stage 2    │    │   Stage 3    │    │   Stage 4    │
│   Network    │───>│   Traffic    │───>│    State     │───>│   Action     │
│  Generation  │    │  Generation  │    │ Observation  │    │  Selection   │
│              │    │              │    │              │    │              │
│ build_       │    │ traffic_     │    │ simulation   │    │ Transformer  │
│ network.py   │    │ gen.py       │    │ .py          │    │ DQN          │
└──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                   │
                    ┌──────────────┐    ┌──────────────┐    ┌──────▼───────┐
                    │   Stage 7    │    │   Stage 6    │    │   Stage 5    │
                    │   Testing    │<───│   Learning   │<───│   Reward     │
                    │  Evaluation  │    │  Experience  │    │ Computation  │
                    │              │    │   Replay     │    │              │
                    │ testing_     │    │ memory.py    │    │ MaxPressure  │
                    │ main.py      │    │ model.py     │    │ + FuzzyLogic │
                    └──────────────┘    └──────────────┘    └──────────────┘
```

---

### Stage 1 — Network Generation

**File:** `build_network.py`

Creates the physical road network that SUMO will simulate on.

```
         N1              N2              N3
         |               |               |
  W ---- TL1 --------- TL2 --------- TL3 ---- E
         |               |               |
         S1              S2              S3
```

**What happens:**
1. Defines **11 nodes** — 3 traffic light intersections (TL1, TL2, TL3) + 8 dead-end entry/exit points
2. Defines **16 edges** — each with 4 lanes, 750m long, speed limit 50 km/h
3. Runs SUMO's `netconvert` tool to generate the binary `environment.net.xml` network file

**Key design decisions:**
- Intersections are spaced 750m apart (realistic urban corridor)
- 4 lanes per direction allows for straight + turning movements
- `netconvert` auto-generates traffic light programs and turn connections

> This is a **one-time setup** step. The network file is reused for all training and testing.

---

### Stage 2 — Traffic Generation

**File:** `simulation/traffic_gen.py`

For each episode, generates a new realistic traffic demand file with randomized vehicle routes.

**Traffic distribution:**

| Category | Probability | Example Routes |
|----------|------------|----------------|
| **Straight traffic** | 75% | W↔E, N1↔S1, N2↔S2, N3↔S3 |
| **Turning traffic** | 25% | W→N1, E→S3, N1→E, S3→W |

**Why Weibull distribution?**
- Vehicle arrival times follow a `Weibull(shape=2)` distribution — not uniform
- This creates a natural **rush-hour effect**: most vehicles arrive early, then traffic tapers off
- The agent must learn to handle both **peak congestion** and **low-traffic recovery**

```
      Traffic Intensity
  ▲
  │   ████
  │  ██████
  │ ████████
  │██████████
  │████████████
  │██████████████████
  └──────────────────────► Time (2700 steps)
       peak        taper
```

**16 distinct routes** are defined across the 3-intersection corridor, including through-traffic (W→E spanning all 3 intersections), cross-traffic (N2→S2 at a single intersection), and turning movements.

---

### Stage 3 — State Observation

**File:** `simulation/simulation.py` → `_get_state()`

The agent perceives the entire traffic situation as a **240-dimensional binary vector**.

```
TraCI API → Get every car's lane + position → Map to 24 lane groups → 
Discretize distance into 10 cells → 240-cell binary occupancy grid
```

**How it works:**
1. Query SUMO (via TraCI) for every vehicle's lane ID and position
2. Map each vehicle to one of **24 lane groups** (8 per intersection: 4 directions × straight/turn)
3. Discretize each vehicle's distance from the stop line into one of **10 cells**
4. Set `state[lane_group × 10 + cell] = 1` for occupied cells

**Cell discretization (non-uniform spacing):**

| Cell | Distance from Stop Line | Resolution |
|------|------------------------|------------|
| 0 | 0 – 7m | 7m *(finest — decisions matter most here)* |
| 1 | 7 – 14m | 7m |
| 2 | 14 – 21m | 7m |
| 3 | 21 – 28m | 7m |
| 4 | 28 – 40m | 12m |
| 5 | 40 – 60m | 20m |
| 6 | 60 – 100m | 40m |
| 7 | 100 – 160m | 60m |
| 8 | 160 – 400m | 240m |
| 9 | 400 – 750m | 350m *(coarsest)* |

> **Design insight:** Non-uniform spacing gives the agent **finer perception near the stop line** (where congestion builds) and coarser perception far away (where exact position matters less). This is a key advantage over flat feature vectors used in traditional ML.

**Final output:** `[0, 0, 1, 0, 0, 1, ..., 0, 0]` — a 240-element binary vector.

---

### Stage 4 — Action Selection

**File:** `simulation/simulation.py` → `_choose_action()`

The Transformer DQN picks one of **8 joint actions** using ε-greedy exploration.

```
             ┌─ Random < ε ?
             │
  State ─────┤  YES → Random action (0–7)     [Exploration]
  (240-dim)  │
             │  NO  → DQN predicts 8 Q-values  [Exploitation]
             │        → argmax(Q) = best action
             └─────────────────────────────────
                     │
                     ▼
              Decode into per-intersection phases
```

**Action decoding (base-2 decomposition):**

| Action | Binary | TL1 Phase | TL2 Phase | TL3 Phase |
|--------|--------|-----------|-----------|-----------|
| 0 | 000 | NS-green | NS-green | NS-green |
| 1 | 001 | NS-green | NS-green | EW-green |
| 2 | 010 | NS-green | EW-green | NS-green |
| 3 | 011 | NS-green | EW-green | EW-green |
| 4 | 100 | EW-green | NS-green | NS-green |
| 5 | 101 | EW-green | NS-green | EW-green |
| 6 | 110 | EW-green | EW-green | NS-green |
| 7 | 111 | EW-green | EW-green | EW-green |

**Exploration schedule (ε-greedy):**
```
ε = 1.0 − (episode / total_episodes)
```
- Episode 1: ε ≈ 1.0 — almost pure random exploration
- Episode 50: ε = 0.5 — balanced exploration/exploitation
- Episode 100: ε = 0.0 — pure exploitation of learned policy

**Phase transitions:**
- If the action changes from the previous step, a **4-second yellow phase** is inserted first
- Yellow is **selectively applied** — only to intersections that actually switch phase, not all three

---

### Stage 5 — Reward Computation

**Files:** `simulation/simulation.py`, `simulation/strategy.py`, `models/fuzzy_logic.py`

The reward signal combines two complementary components to guide the agent's learning.

#### Component A — Wait-Time Delta Reward

```python
wait_reward = old_total_wait - current_total_wait
```

- Tracks accumulated waiting time of every car on all **12 incoming edges**
- **Positive** when total waiting time decreases (good action)
- **Negative** when total waiting time increases (bad action)
- Cars that clear an intersection are removed from tracking

#### Component B — Fuzzy Logic Congestion Penalty

The fuzzy inference system evaluates overall network congestion using two inputs:

**Inputs:**
- `density` (0–1): average vehicle density across all incoming roads (count / capacity)
- `queue` (0–49): total number of halting vehicles

**Fuzzy Rules:**
| Rule | Condition | → Congestion Level |
|------|-----------|-------------------|
| 1 | density=low AND queue=short | → **Low** |
| 2 | density=low AND queue=medium | → **Medium** |
| 3 | density=medium AND queue=short | → **Medium** |
| 4 | density=medium AND queue=medium | → **Medium** |
| 5 | density=high OR queue=long | → **High** |

**Output:** congestion score (0.0 to 1.0)

#### Combined Reward

```
Final Reward = wait_reward − (congestion_score × 0.5)
```

The `× 0.5` multiplier keeps the fuzzy penalty small relative to the wait-time signal, so it acts as a **gentle directional nudge** rather than overwhelming the primary learning signal.

---

### Stage 6 — Learning

**Files:** `models/model.py`, `utils/memory.py`, `simulation/simulation.py` → `_replay()`

After each episode, the agent trains on sampled past experiences.

#### Experience Replay Buffer

| Property | Value |
|----------|-------|
| Max capacity | 50,000 transitions |
| Min before training | 600 transitions |
| Eviction policy | FIFO (oldest removed first) |
| Sampling method | Uniform random (breaks temporal correlations) |

Each transition stored: `(state, action, reward, next_state, done)`

#### Q-Learning Update (Bellman Equation)

```
Q(s, a) ← reward + γ × max_a' Q(s', a')
```

For terminal states (end of episode): `Q(s, a) = reward`

#### Training Loop

1. Episode ends → **8 training epochs** begin
2. Each epoch: sample **32 random transitions** from the buffer
3. Compute Q-targets using the Bellman equation (γ = 0.75)
4. Loss = MSE between predicted Q-values and targets
5. Backpropagate through the Transformer + Dense layers
6. Adam optimizer updates weights (learning rate = 0.001)

---

### Stage 7 — Testing & Evaluation

**Files:** `testing_main.py`, `testing_simulation.py`

Loads the saved model and runs a single episode with **no exploration** (pure exploitation).

**What it measures:**
- Total vehicles that passed through the network
- Average waiting time per vehicle (seconds)
- Average queue length (vehicles)
- Total reward accumulated
- Action distribution (how often each of the 8 actions was chosen)

**Output:**
- Efficiency metrics saved to `efficiency_metrics.txt`
- Reward and queue plots saved as PNG images
- SUMO-GUI opens for visual verification

---

## Model Architecture

The neural network uses a **Transformer Encoder** to learn inter-lane relationships, followed by dense layers for Q-value prediction.

```
Input (240)
  │
  ▼
Reshape → (24 tokens × 10 features)
  │         Each token = 1 lane group
  ▼
Multi-Head Self-Attention (2 heads, key_dim=10)
  │         Learns which lanes should coordinate
  ▼
Residual Connection + LayerNorm
  │
  ▼
Feed-Forward Network (256 → 10) + Residual + LayerNorm
  │
  ▼
Flatten → 240
  │
  ▼
Dense (256 units, ReLU) + Dropout (0.1)
  │
  ▼
Dense (256 units, ReLU) + Dropout (0.1)
  │
  ▼
Output → 8 Q-values (one per joint action)
```

**Why Transformer instead of plain Dense layers?**

Each of the 24 lane groups becomes a "token" in the Transformer. The self-attention mechanism learns **which lanes should coordinate with which** — for example, TL1's east-bound queue and TL2's west-bound queue are directly connected and must be considered together. A plain MLP treats all 240 inputs as independent, missing these structural relationships.

---

## Why Deep RL Over Traditional ML

### The Core Problem

Traffic signal control is a **sequential decision-making** problem under uncertainty. Each phase choice affects all future traffic states. Traditional ML treats each decision independently — Deep RL treats it as a **continuous optimization process** over time.

### Comparison Table

| Dimension | Traditional ML (Supervised / Rule-Based) | This Deep RL Approach | Advantage |
|-----------|----------------------------------------|----------------------|-----------|
| **Learning Source** | Requires labeled datasets (state → optimal action) | Learns from interaction with environment (trial-and-error) | No labels needed |
| **Adaptability** | Static model; fails on unseen traffic patterns | Continuously discovers new strategies through exploration | Handles novel situations |
| **Multi-Intersection Coordination** | Each intersection optimized independently (greedy) | Single agent jointly controls all 3 intersections | Learns coordination |
| **Temporal Reasoning** | Ignores that current action affects future states | Bellman equation models future consequences (γ = 0.75) | Plans ahead |
| **Feature Engineering** | Requires manual feature design by domain experts | Self-attention automatically discovers lane relationships | Self-learned features |
| **Objective Alignment** | Optimizes a proxy metric (e.g., classification accuracy) | Directly optimizes the real goal (minimize total wait) | Aligned objectives |
| **Non-Stationarity** | Assumes fixed data distribution | Handles changing traffic demand (Weibull rush-hour) | Robust to change |
| **Expert Knowledge** | Hard to incorporate domain rules | Fuzzy Logic integrates expert knowledge as reward shaping | Best of both worlds |
| **Training Data** | Needs massive labeled datasets | Generates own data through simulation | Self-sufficient |
| **Scalability** | N independent models for N intersections | One model handles N intersections jointly | Scales gracefully |

### vs. Specific Traditional Approaches

| Approach | Limitation | How This System Overcomes It |
|----------|-----------|------------------------------|
| **Fixed-Timer Control** | Preset cycle lengths can't adapt to demand | Agent learns demand-responsive policies |
| **Actuated Control (SCATS/SCOOT)** | Reactive only — extends/skips phases based on detectors | Agent uses predictive Q-values for future congestion |
| **Webster's Formula / TRANSYT** | Assumes steady-state traffic, optimizes offline | Agent handles non-stationary Weibull patterns in real-time |
| **Supervised Deep Learning** | Needs "optimal action" labels — which don't exist | Self-supervised through reward signals |
| **Simple DQN (no Transformer)** | Treats 240 inputs as flat vector, no structural awareness | Self-attention learns inter-lane spatial dependencies |

### Why Each Component Matters

| Component | Traditional Equivalent | Advantage of This Approach |
|-----------|----------------------|---------------------------|
| **Deep Q-Learning** | Supervised classifier | Learns by trial-and-error; directly maximizes the real objective |
| **Transformer Encoder** | Manual feature engineering | Self-attention automatically discovers which lanes need to coordinate |
| **Fuzzy Logic Reward** | Hard-coded thresholds | Smooth, interpretable, handles uncertainty gracefully |
| **Experience Replay Buffer** | Online gradient descent | Breaks temporal correlations; reuses rare events for stability |

---

## Training Results

### Hyperparameters (Model 15 — Latest)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `total_episodes` | 100 | Training episodes |
| `max_steps` | 2700 | Simulation steps per episode (~45 min simulated) |
| `n_cars_generated` | 600 | Vehicles per episode |
| `green_duration` | 20s | Green phase duration |
| `yellow_duration` | 4s | Yellow transition duration |
| `batch_size` | 32 | Experience replay batch size |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `training_epochs` | 8 | Replay passes per episode |
| `gamma` | 0.75 | Discount factor |
| `memory_size_min` | 600 | Min samples before training starts |
| `memory_size_max` | 50,000 | Replay buffer capacity |
| `num_layers` | 2 | Dense layers after Transformer block |
| `width_layers` | 256 | Neurons per dense layer |

### Training Performance (100 Episodes)

| Metric | Episode 1 | Episode 50 | Episode 100 | Improvement |
|--------|-----------|------------|-------------|-------------|
| **Cumulative Reward** | -21.59 | -18.88 | -18.30 | 15.2% better |
| **Cumulative Delay (s)** | 19,731 | 13,513 | 7,908 | **59.9% reduction** |
| **Avg Queue Length** | 7.31 | 5.00 | 2.93 | **59.9% reduction** |

**Key observations:**
- Cumulative delay dropped from **19,731s to 7,908s** — a **59.9% reduction** over 100 episodes
- Average queue length dropped from **7.31 to 2.93 vehicles** — also a **59.9% reduction**
- The reward signal stabilized around episode 40, indicating policy convergence
- Two spike episodes (23 and 77) show occasional exploration instability, but the agent recovers immediately

---

## Getting Started

### Prerequisites

- Python 3.11+
- [SUMO Traffic Simulator](https://sumo.dlr.de/docs/Downloads.php) (1.20+)
- SUMO_HOME environment variable set

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Traffic-AI-Project

# Install dependencies
pip install -r requirements.txt
```

### Generate the 3-Intersection Network

```bash
python build_network.py
```

This generates `intersection/environment.net.xml` with the 3-intersection corridor layout using SUMO's `netconvert`.

### Train the Agent

```bash
python main_train.py
```

Training configuration is in `training_settings.ini`. Models are saved to `models/model_<N>/`.

### Test & Visualize

```bash
python testing_main.py
```

Opens the **SUMO GUI** to visually watch the trained agent control all 3 intersections. Test results (plots + metrics) are saved to `models/model_<N>/test/`.

> **Note:** Update `model_to_test` in `testing_settings.ini` to match your latest model number.

---

## Project Structure

```
Traffic-AI-Project/
├── main_train.py              # Training entry point
├── testing_main.py            # Testing entry point with SUMO GUI
├── testing_simulation.py      # Test simulation (3-TL, SUMO GUI)
├── build_network.py           # Generates 3-intersection SUMO network
├── training_settings.ini      # Training hyperparameters
├── testing_settings.ini       # Testing configuration
├── requirements.txt           # Python dependencies
├── yolo11n.pt                 # Pre-trained YOLO model for perception
│
├── simulation/
│   ├── simulation.py          # Training simulation (joint 3-TL control)
│   ├── traffic_gen.py         # Route file generator (Weibull distribution)
│   └── strategy.py            # Max Pressure reward strategy
│
├── models/
│   ├── model.py               # Transformer-based DQN (Train + Test classes)
│   ├── fuzzy_logic.py         # Fuzzy congestion evaluator (scikit-fuzzy)
│   ├── perception.py          # YOLO perception module (real-world bridge)
│   └── model_<N>/             # Saved models + training plots
│       ├── trained_model.h5
│       ├── plot_reward.png
│       ├── plot_delay.png
│       ├── plot_queue.png
│       └── training_settings.ini
│
├── intersection/
│   ├── environment.net.xml    # SUMO network file (auto-generated)
│   ├── sumo_config.sumocfg    # SUMO configuration
│   ├── episode_routes.rou.xml # Per-episode route file (auto-generated)
│   ├── nodes.nod.xml          # Node definitions
│   ├── edges.edg.xml          # Edge definitions
│   └── tll.tll.xml            # Traffic light logic
│
├── utils/
│   ├── memory.py              # Experience replay buffer
│   ├── visualization.py       # Matplotlib plotting
│   └── utils.py               # Config parsing & path management
│
└── tests/                     # Test suite
```

---

## Complete Data Flow (One Training Step)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SUMO SIMULATOR                               │
│  ┌──────┐    ┌──────┐    ┌──────┐                                  │
│  │ TL1  │────│ TL2  │────│ TL3  │   600 vehicles, 16 route types   │
│  └──────┘    └──────┘    └──────┘                                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │ TraCI API
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STATE ENCODER                                                      │
│  • Loop through all vehicles in simulation                         │
│  • Map each to (lane_group, distance_cell)                         │
│  • Produce 240-bit binary occupancy vector                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ACTION SELECTOR (ε-greedy)                                         │
│  • If random < ε → random action (0–7)           [exploration]     │
│  • Else → Transformer DQN → argmax(Q-values)     [exploitation]   │
│  • Decode: action 5 → (TL1=EW, TL2=NS, TL3=EW)                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE EXECUTION                                                    │
│  1. If phase changed → set yellow (4s) on changed intersections    │
│  2. Set green phase on all 3 intersections (20s)                   │
│  3. SUMO advances 20–24 simulation steps                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  REWARD COMPUTATION                                                 │
│  wait_reward    = old_total_wait − current_total_wait              │
│  fuzzy_score    = FuzzyLogic(avg_density, total_queue) → [0, 1]    │
│  ──────────────────────────────────────────────────                 │
│  final_reward   = wait_reward − (fuzzy_score × 0.5)               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  EXPERIENCE REPLAY BUFFER                                           │
│  Store: (state, action, reward, next_state, done)                  │
│  Capacity: 50,000 │ Min before training: 600                      │
└────────────────────────────┬────────────────────────────────────────┘
                             │ (after episode ends)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TRAINING (8 epochs × batch of 32)                                  │
│  1. Sample 32 random transitions from buffer                       │
│  2. Q_target = reward + 0.75 × max(Q(next_state))                 │
│  3. Loss = MSE(Q_predicted, Q_target)                              │
│  4. Backpropagate through Transformer + Dense layers               │
│  5. Adam optimizer updates weights (lr = 0.001)                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## References

- Andrea Vidali — University of Milano-Bicocca (original single-intersection framework)
- "Deep Reinforcement Learning for Traffic Light Control in Intelligent Transportation Systems" — Xiao-Yang Liu, Ming Zhu, Sem Borst, Anwar Walid
- Max Pressure Control — Varaiya, P. (2013). "Max pressure control of a network of signalized intersections"
- Fuzzy Logic — scikit-fuzzy library for congestion inference
- Transformer Architecture — Vaswani et al. (2017). "Attention Is All You Need"
- SUMO Documentation — [https://sumo.dlr.de/docs/](https://sumo.dlr.de/docs/)
