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
- 🔄 TorchScript (JIT) or ONNX export for deployment

---
## Results 👀

Following Results were achieved using this setup:
- Intel Core i5-13600K (14 cores, 20 threads)
- 32 GB RAM (DDR5 - 6000 MT/s)
- NVIDIA GeForce RTX 4070 Ti
- 4096 Envs, 1500 iterations -> runtime: 45min

Dodo:

https://github.com/user-attachments/assets/8276639f-ffef-4358-a578-7138b7d5190d

<img width="1658" height="715" alt="image" src="https://github.com/user-attachments/assets/655c3ed9-ef01-461f-8c61-9e362383d8f7" />
<img width="2469" height="1260" alt="image" src="https://github.com/user-attachments/assets/0e6ed3f3-12a9-408f-8176-a8bbe92f4dab" />


GO2:

https://github.com/user-attachments/assets/afaef573-847f-4afc-a96d-eb92982bcb41

<img width="1657" height="677" alt="image" src="https://github.com/user-attachments/assets/d4dcaaed-5765-42fa-9461-c76f04ecce89" />
<img width="2469" height="1258" alt="image" src="https://github.com/user-attachments/assets/2ecb48ee-5110-4de5-927c-b43a35b0307a" />


---

## 🔁 Pipeline Overview

- **Train**: Learn policy using PPO in simulation  
- **Play**: Visualize trained policy
- **Export**: Export a trained policy in order to deploy it (Sim2Sim or Sim2Real)

---

## 📁 Project Structure

```text
robot_gym/
│
├── envs/
│ ├── base/
│ ├── dodo/
│ ├── go2/
│
├── scripts/
│ ├── train.py
│ ├── play.py
│
├── utils/
│ ├── task_registry.py
│ ├── helpers.py
│ ├── debug.py
│ ├── math.py
│ ├── terrain.py
│ ├── urdf_reader.py
│
├── ressources/
│ ├── robots/
│ ├── pretrained/
logs/

```

-> The main structure and many design choices are based on this repository: https://github.com/unitreerobotics/unitree_rl_gym/tree/main

---

## ⚙️ Installation

### 1. Clone

SSH example:

```bash
git clone git@github.com:liambansbach/bipedal_robot_sim.git
cd bipedal_robot_sim
```

### 2. Setup environment

```bash
conda env create -f conda_env.yaml
conda activate rl-genesis
```

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
  - If you want to train your own robot, simply add your URDF file to "ressources/robots/", create a new config and env file in "robot_gym/envs" and also register the new task in "envs/__init__.py".

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

### 🧠 Observations

Typical observation vector:

- base linear velocity
- base angular velocity
- projected gravity
- command velocities
- joint positions
- joint velocities
- previous actions

### 🎯 Reward System

Modular reward design:

- Base rewards
- velocity tracking
- orientation stability
- base height
- smoothness penalties
- Robot-specific rewards
- foot swing clearance
- flat feet
- torso pitch
- hip penalties (avoid clinching legs together)
- survival reward

Implemented in LeggedRobot env and robot specific env (e.g.: DodoEnv)

### ⚙️ Configuration System

Hierarchical config system:

- EnvCfg → simulation + robot
- RewardCfg → reward shaping
- TrainCfg → PPO

Configs auto-instantiate recursively

### 🤖 Automatic Robot Parsing

- extracts joint names
- resolves paths automatically

Implemented via URDFReader

You can easily use your own URDF robot file for training your own locomotion policy. Just make sure that its consistent with the provided pipeline and that the URDF is optimized. Optimizing can include "simplifying collisions", by using collision-boxes or cylinders instead of the actual meshes. This will reduce training time by a lot.

### 📊 Logging

- Weights & Biases integration OR Tensorboard (as given by RSL-RL)
- reward breakdown
- training metrics

---

## 🧪 Example Commands

```bash
# Full training
python -m robot_gym.scripts.train --task=dodo --num_envs 4096
# Debug run
python -m robot_gym.scripts.train --task=dodo --num_envs 512 --max_iterations 50
# Play model
python -m robot_gym.scripts.play --task=dodo
```

---

## 🔮 Future Work

This pipeline is still at an early stage and will be extended from time to time with the following functionalities:

- two-stage training (Teacher–Student Learning with Privileged Information)
- further domain randomization
- curriculum learning
- sim2real (ROS2)

---

## Contact

If you encounter any issues, feel free to contact me:

mail: liam.bansbach@tum.de
