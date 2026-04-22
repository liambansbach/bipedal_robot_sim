# RL GYM – Reinforcement Learning for Legged Robots (Genesis)

This repository provides a **modular reinforcement learning pipeline** for training legged robots using the **Genesis physics engine**.

It is inspired by the structure of Unitree RL pipelines but fully adapted to:

- ⚡ **Genesis (GPU accelerated physics)**
- 🤖 **Custom robots (URDF / MJCF)**
- 🧠 **PPO (rsl-rl)**

---

## 🚀 Features

- ⚡ GPU-accelerated RL training with **Genesis**
- 🧠 PPO implementation via **rsl-rl**
- 🏗️ Clean **task + config + registry system**
- 🤖 Automatic **URDF parsing (joints + feet)**
- 🔁 Supports **Train → Play → Export → Deployment**
- 📊 Integrated logging via **Weights & Biases**
- 🌍 Multiple environments (flat + uneven terrain)
- 🔄 TorchScript (JIT) export for deployment

---

## 🔁 Pipeline Overview

- **Train**: Learn policy using PPO in simulation  
- **Play**: Visualize trained policy  
- **Sim2Sim**: Transfer to other simulators  
- **Sim2Real**: Deploy to hardware (future)

---

## 📁 Project Structure

robot_gym/
│
├── envs/
│ ├── base/
│ ├── dodo/
│
├── scripts/
│ ├── train.py
│ ├── play.py
│
├── utils/
│ ├── task_registry.py
│ ├── helpers.py
│ ├── logger.py
│ ├── math.py
│ ├── urdf_reader.py
│
logs/
ressources/robots/

---

## ⚙️ Installation

### 1. Clone

```bash
git clone <your-repo>
cd <your-repo>
```

### 2. Setup environment

```bash
conda env create -f conda_env.yaml
conda activate rl-genesis
```

⚠️ Important:
pip install rsl-rl-lib==2.2.4
-> This version works stable with the pipeline. Others might not work!

---

## 🛠️ Usage

### 🏋️ Training

```bash
python -m robot_gym.scripts.train --task dodo --experiment_name dodo_walking_test --num_envs 4096 --max_iterations 1000
```

Training pipeline:

- PPO via rsl-rl
- parallel environments on GPU
- config-driven setup via TaskRegistry

### 👀 Play (Evaluation)

```bash
python -m robot_gym.scripts.play --task=dodo
```

- loads latest checkpoint
- runs inference policy
- exports JIT automatically

### 💾 Export Policy

Saved automatically to:

```bash
logs/<experiment>/exported/policies/policy_1.pt
```

Export uses TorchScript for deployment

### 🧠 Observations

Typical observation vector (~36D):

- base linear velocity
- base angular velocity
- projected gravity
- command velocities
- joint positions
- joint velocities
- previous actions

Defined in base environment

### 🎯 Reward System

Modular reward design:

- Base rewards
- velocity tracking
- orientation stability
- base height
- smoothness penalties
- Dodo-specific rewards
- foot swing clearance
- flat feet
- torso pitch
- hip penalties
- survival reward

Implemented in DodoEnv

### ⚙️ Configuration System

Hierarchical config system:

- EnvCfg → simulation + robot
- RewardCfg → reward shaping
- TrainCfg → PPO

Configs auto-instantiate recursively

### 🤖 Automatic Robot Parsing

No manual joint mapping required.

- extracts joint names
- detects foot links
- resolves paths automatically

Implemented via URDFReader

### 📊 Logging

- Weights & Biases integration
- reward breakdown
- training metrics

Integrated in training script

---

## 🧪 Example Commands

```bash
# Full training
python robot_gym/scripts/train.py --task=dodo --num_envs 4096
# Debug run
python robot_gym/scripts/train.py --task=dodo --num_envs 512 --max_iterations 50
# Play model
python robot_gym/scripts/play.py --task=dodo
```

---

## 🔮 Future Work

- sim2real (ROS2)
- domain randomization
- curriculum learning
- advanced terrain
