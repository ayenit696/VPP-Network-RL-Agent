# ⚡ Autonomous Virtual Power Plant (VPP) Orchestrator

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Reinforcement Learning](https://img.shields.io/badge/RL-Stable--Baselines3-orange)
![Grid Physics](https://img.shields.io/badge/Physics-Gym--ANM-success)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 🌍 The Problem: The "Messy" Reality of the Grid
As the world shifts to renewable energy, modern power grids rely on decentralized assets—solar inverters, wind turbines, and electric vehicle (EV) batteries. A **Virtual Power Plant (VPP)** aggregates these distributed assets to act as a single massive battery. 

However, orchestrating thousands of devices over cellular or IoT networks introduces severe communication constraints: **packet loss, jitter, and latency**. If an ML model sends a command to discharge batteries to stabilize grid frequency, but the network delays that packet, the AI risks over-correcting and destabilizing the entire physical grid.

## 🧠 The Solution: Robust Reinforcement Learning
This project implements a **Deep Reinforcement Learning (DRL)** framework engineered explicitly to stabilize AC/DC grid power flows under degraded, unstable network environments. 

Rather than assuming a perfect, instantaneous communication layer, this architecture forces the AI to operate as a **Partially Observable Markov Decision Process (POMDP)**. By utilizing Proximal Policy Optimization (PPO) with an expanded neural network architecture, the agent develops internal "belief states" to calculate the uncertainty of stale telemetry data and safely bid energy without breaching physical thermal line limits.

---

## 🚀 Key Technical Features

* **Real AC/DC Power Flow Physics:** Replaced standard toy-math environments with `gym-anm`, simulating real-world electrical engineering constraints, active/reactive power injections, and local transformer thermal limits.
* **Custom Network Emulation Layer:** Engineered a native `Gymnasium Wrapper` that programmatically injects stochastic network degradation:
  * **Packet Drops:** Simulates 15% random cellular data loss (telemetry freezing).
  * **Dynamic Jitter:** Introduces asynchronous data arrival times.
* **Legacy-to-Modern AI Bridging:** Utilized `shimmy` translation layers to bridge legacy physics engines with modern `Stable-Baselines3` ML architectures.

---

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `vpp_env.py` | The foundational custom environment testing basic latency math. |
| `train_vpp.py` | Training script for the foundational delay model. |
| `network_wrapper.py` | **Core Feature:** The middleware injecting stochastic packet drops and jitter. |
| `train_anm.py` | Training script merging complex AC/DC physics with the messy network. |
| `test_anm.py` | **Execution:** Live real-time inference and evaluation script. |
| `vpp_anm_robust_model.zip` | The fully trained, robust PPO neural network weights. |

---

## 💻 Installation & Quick Start

Want to see the AI save the grid from a blackout in real-time? 

**1. Clone the repository**
```bash
git clone [https://github.com/YourUsername/VPP-Network-RL-Agent.git](https://github.com/ayenit696/VPP-Network-RL-Agent.git)
cd VPP-Network-RL-Agent
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Live Simulation**
```bash
python test_anm.py
```


## 📊 Live Simulation Results
Below is the verified diagnostic logging sequence generated from test_anm.py.
Notice how the AI maintains smooth control stability even during sudden physical grid shocks combined with complete connectivity drops (❌ DATA LOST). Instead of panicking and throwing anomalous commands that overload lines, the model uses its predictive momentum to balance voltage safety boundaries perfectly.

---
 ⚡ INITIATING LIVE GRID ORCHESTRATION ⚡
---

|Step 20 | Network: ✅ Telemetry Delivered            | Grid Status Reward: -0.08|

Step 21 | Network: ✅ Telemetry Delivered            | Grid Status Reward: -0.07

Step 22 | Network: ✅ Telemetry Delivered            | Grid Status Reward: -0.06

Step 23 | Network: ✅ Telemetry Delivered            | Grid Status Reward: -0.77 (Physical Grid Shock)

Step 24 | Network: ✅ Telemetry Delivered            | Grid Status Reward: -0.92

Step 25 | Network: ✅ Telemetry Delivered            | Grid Status Reward: -2.12 (Corrective Action Taken)

Step 26 | Network: ❌ DATA LOST (Using Stale State)  | Grid Status Reward: -1.83 (AI coasts blindly)

Step 27 | Network: ✅ Telemetry Delivered            | Grid Status Reward: -1.83

Step 28 | Network: ✅ Telemetry Delivered            | Grid Status Reward: -1.46 (Grid stabilized)

Step 29 | Network: ✅ Telemetry Delivered            | Grid Status Reward: -1.42

Step 30 | Network: ✅ Telemetry Delivered            | Grid Status Reward: -1.40

--- Simulation Complete ---
🏆 SUCCESS: The AI successfully maintained grid stability without melting the power lines!