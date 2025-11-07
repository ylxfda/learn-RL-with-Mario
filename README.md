# 🧠 Learn Reinforcement Learning with Mario

Welcome to **Learn Reinforcement Learning with Mario** — an educational repository that teaches you the evolution of **policy gradient reinforcement learning (RL)** algorithms through the lens of the **Super Mario** game.

This repository contains **PyTorch implementations** of:
- ✅ **PPO (Proximal Policy Optimization)**
- ✅ **DreamerV3 (World Model–based RL)**

The goal of this README is to guide you from **zero RL background** to understanding **how and why** modern RL algorithms were designed — by answering a series of practical, question-driven learning steps.

---

## 🎮 Why Mario?

Mario’s environment is the perfect playground for understanding RL concepts:
- He sees the world (state).
- He chooses actions (move, jump, run).
- He receives feedback (reward).
- He learns to **maximize long-term success**.

---

## 🧩 What Is Reinforcement Learning?

Reinforcement Learning (RL) teaches an agent how to act by interacting with an environment.

At each time step:
- The agent observes a **state** \( s_t \),
- Takes an **action** \( a_t \),
- Receives a **reward** \( r_t \),
- And transitions to the next state \( s_{t+1} \).

The goal is to learn a **policy** \( \pi_\theta(a|s) \) — a mapping from states to actions — that maximizes the **expected cumulative reward**:

\[
J(\theta) = \mathbb{E}_{\pi_\theta} \Big[ \sum_{t=0}^\infty \gamma^t r_t \Big]
\]

---

## 🪜 Step-by-Step Evolution of RL Algorithms (with Mario Examples)

Below we explore the major milestones — from **REINFORCE** to **PPO**, and finally to **DreamerV3** — always asking:

> 💡 What problem are we solving at each step?

---

### 1️⃣ REINFORCE — *“Can Mario learn just from rewards?”*

**Idea:**  
After each episode, update the policy based on total reward.

**Update rule:**
\[
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a_t|s_t) R_t] \tag{1}
\]

**Intuition:**  
If a sequence of actions leads to a high reward, make those actions more likely.

**Mario Example:**  
Mario randomly jumps; if he survives longer, reinforce those jumps.

**Problem:**  
- Very high variance — results change wildly from one episode to another.  
- Learns slowly — feedback only comes at the end.

---

### 2️⃣ Add a Baseline — *“Can Mario judge actions relative to his usual performance?”*

**Idea:**  
Subtract a baseline value \( V^\pi(s_t) \) representing expected performance to reduce noise.

**Update rule:**
\[
\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a_t|s_t)(R_t - V^\pi(s_t))] \tag{2}
\]

**Advantage function:**
\[
A_t = R_t - V^\pi(s_t)
\]

**Mario Example:**  
If Mario usually earns +5 coins but now earns +10, he learns that this jump was better than usual.

**Benefit:**  
Reduces variance → more stable learning.

---

### 3️⃣ Actor–Critic — *“Can Mario get feedback immediately instead of waiting until he dies?”*

**Idea:**  
Add a **Critic** network to estimate \( V(s_t) \) (the baseline) while the **Actor** updates the policy.

**Temporal-Difference Advantage:**
\[
A_t = r_t + \gamma V(s_{t+1}) - V(s_t) \tag{3}
\]

**Mario Example:**  
Now Mario gets real-time feedback — every frame tells him whether he’s improving or not.

**Benefit:**  
- Online updates (no need for full episodes).  
- Faster, more continuous learning.

---

### 4️⃣ A2C / A3C — *“Can many Marios learn in parallel?”*

**Idea:**  
Run multiple Mario agents simultaneously in parallel environments.  
Each agent collects experiences and contributes gradients to a shared model.

**Benefit:**  
- Faster data collection.  
- Smoother gradient estimation.  
- More stable learning.

---

### 5️⃣ TRPO — *“How can Mario avoid sudden, catastrophic policy changes?”*

**Problem:**  
Even with a critic, large updates can cause the policy to change too drastically.

**Solution:**  
Add a **trust region** constraint — restrict how much the new policy can deviate from the old one.

\[
\begin{aligned}
\max_\theta &\ \mathbb{E}_t\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}A_t\right] \\
\text{s.t. } &\ \mathbb{E}_t[KL(\pi_{\theta_{\text{old}}} \| \pi_\theta)] \le \delta
\end{aligned} \tag{4}
\]

**Mario Example:**  
Mario doesn’t completely change his jumping style overnight; he takes safe, measured steps in learning.

**Drawback:**  
Computationally expensive due to second-order gradient constraints.

---

### 6️⃣ PPO — *“Can we simplify safe updates while keeping them stable?”*

**Idea:**  
Replace the hard constraint of TRPO with an easy-to-compute **clipped surrogate objective**.

\[
L^{CLIP}(\theta) = \mathbb{E}_t \Big[
\min\big(
r_t(\theta)A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t
\big)
\Big] \tag{5}
\]

where  
\[
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
\]

**Mario Example:**  
If Mario’s new policy changes too much (say, more than ±20%), we clip it to keep updates stable.

**Benefits:**  
- Simple implementation.  
- High performance.  
- Stable learning in complex environments.

This is the first algorithm included in this repo.

---

### 7️⃣ DreamerV3 — *“Can Mario imagine before acting?”*

**Problem with PPO:**  
It learns only from **real interactions**, requiring millions of frames.  
Mario must die many times to improve.

**Idea:**  
Teach Mario to **build a world model** — an internal simulation of how the game behaves — and learn by “dreaming” inside it.

---

#### 🧠 Core Components

1. **World Model (Encoder + Transition + Decoder):**  
   Learns to compress observations into latent states \( z_t \) and predict next states, rewards, and continuation signals.

2. **Imagination Rollouts:**  
   Generates imaginary trajectories \( (z_t, a_t, r_t) \) within the latent space instead of the real game.

3. **Actor & Critic in Latent Space:**  
   Uses imagined trajectories to train the policy and value functions efficiently.

---

**Training Loop Overview:**

1. Collect real experiences for a short time.  
2. Train the world model to predict future states.  
3. Use the model to “imagine” many future rollouts.  
4. Optimize policy and value inside the imagined world.  
5. Occasionally update with real experiences.

---

**Mario Example:**  
Mario watches a few rounds of gameplay, learns how the world behaves, and then mentally simulates thousands of jumps, enemy encounters, and coin collections — all in his mind — before trying them in the real game.

**Benefits:**
- Learns from far fewer real frames.  
- Much faster and safer training.  
- Generalizes better.

This is the **second algorithm** implemented in this repo.

---

## 🔁 Summary: Evolution of Mario’s Learning

| Stage | Algorithm | Key Idea | Mario’s Learning Style |
|--------|------------|-----------|------------------------|
| 1️⃣ | REINFORCE | Learn from total reward | Trial and error |
| 2️⃣ | + Baseline | Compare to average | Learns relative success |
| 3️⃣ | Actor–Critic | Add a value estimator | Real-time feedback |
| 4️⃣ | A2C/A3C | Parallel agents | Multiple worlds |
| 5️⃣ | TRPO | Limit policy change | Careful improvement |
| 6️⃣ | PPO | Simplify safe updates | Balanced, efficient learning |
| 7️⃣ | DreamerV3 | Learn a world model | Imagines and plans ahead |

---

## 🧰 Repository Structure

```
Learn-Reinforcement-Learning-with-Mario/
│
├── PPO/
│   ├── ppo_agent.py
│   ├── ppo_train.py
│   └── README.md
│
├── DreamerV3/
│   ├── dreamer_agent.py
│   ├── dreamer_train.py
│   └── README.md
│
├── utils/
│   ├── envs.py      # Mario Gym environment wrappers
│   └── plotting.py  # Visualization helpers
│
└── README.md         # (this file)
```

---

## ⚙️ Getting Started

### Installation
```bash
git clone https://github.com/yourname/Learn-Reinforcement-Learning-with-Mario.git
cd Learn-Reinforcement-Learning-with-Mario
pip install -r requirements.txt
```

### Run PPO Training
```bash
python PPO/ppo_train.py
```

### Run DreamerV3 Training
```bash
python DreamerV3/dreamer_train.py
```

---

## 📘 References

- Williams, R. J. (1992). *Simple statistical gradient-following algorithms for connectionist reinforcement learning.*
- Schulman et al. (2015). *Trust Region Policy Optimization.*
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms.*
- Hafner et al. (2023). *Mastering Diverse Domains through World Models (DreamerV3).*

---

### 💬 Final Thought

> PPO taught Mario to **learn steadily from real experiences.**  
> DreamerV3 taught Mario to **think and plan inside his own imagination.**

---

**Enjoy exploring, modifying, and training your own Mario agent!**
