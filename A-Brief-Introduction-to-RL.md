# 🪜 A Brief Introduction to Reinforcement Learning

This document provides a self-contained introduction to reinforcement learning, tracing the evolution from simple policy gradients to modern world models. No prior RL knowledge required!

---

## 📋 Table of Contents

- [What Is Reinforcement Learning?](#-what-is-reinforcement-learning)
- [Why Mario?](#-why-mario)
- [The Evolution of RL Algorithms](#the-evolution-of-rl-algorithms-a-question-driven-journey)
  - [1️⃣ REINFORCE](#1️⃣-reinforce-can-we-learn-directly-from-rewards)
  - [2️⃣ Baseline](#2️⃣-baseline-how-do-we-reduce-the-noise)
  - [3️⃣ Actor-Critic](#3️⃣-actor-critic-can-we-learn-online-step-by-step)
  - [4️⃣ A2C/A3C](#4️⃣-a2ca3c-what-if-many-marios-learn-simultaneously)
  - [5️⃣ TRPO](#5️⃣-trpo-how-do-we-ensure-safe-gradual-improvement)
  - [6️⃣ PPO](#6️⃣-ppo-can-we-simplify-safe-updates)
  - [7️⃣ DreamerV3](#7️⃣-dreamerv3-can-we-learn-by-dreaming)
- [Summary Table](#-summary-the-journey-from-reinforce-to-dreamerv3)
- [References](#-references-and-further-reading)

---

## 🧩 What Is Reinforcement Learning?

**The Core Idea:**
Reinforcement learning teaches an agent to make decisions through trial and error, guided by rewards.

**The Loop:**
At each time step $t$:

1. Agent observes **state** $s_t$
2. Agent selects **action** $a_t$ based on its policy $\pi_\theta(a|s)$
3. Environment returns **reward** $r_t$ and next **state** $s_{t+1}$
4. Agent updates its policy to get better rewards in the future

**The Objective:**
Learn a policy $\pi_\theta$ that maximizes **expected cumulative reward**:

$$
J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right] \tag{1}
$$

where $\gamma \in [0,1)$ is the discount factor (future rewards matter less than immediate ones).

---

## 🎮 Why Mario?

Mario is the perfect playground for learning RL because his world contains all the essential elements:

- **States**: What Mario observes (enemies, blocks, pipes, terrain)
- **Actions**: What Mario can do (move left/right, jump, run, crouch)
- **Rewards**: What Mario receives (coins, points for defeating enemies, reaching the flag)
- **Goal**: Learn a strategy that maximizes long-term success

This simple yet rich environment lets you focus on understanding RL algorithms without getting lost in complex domain details.

---

## The Evolution of RL Algorithms: A Question-Driven Journey

Let's trace the path from simple policy gradient methods to modern world models. At each stage, we'll ask: **"What's wrong with what we have, and how can we fix it?"**

---

### 1️⃣ REINFORCE: *Can we learn directly from rewards?*

**The Starting Point:**
What if Mario just tries random actions, and after each episode, we make actions that led to good outcomes more likely?

**How it works:**
After completing an episode, compute the total return $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ from each time step, then update:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a_t|s_t) \, G_t] \tag{1}
$$

**Intuition:**
If a sequence of actions led to high reward, increase their probability. If it led to low reward, decrease it.

**Mario's Experience:**
Mario tries jumping randomly. If he survives longer in one episode, he reinforces those specific jumps.

**❌ The Problem:**
- **High variance**: Results swing wildly between episodes
- **Slow learning**: Feedback only comes at the end of entire episodes
- **Credit assignment**: Hard to know *which* actions were actually good

> 🤔 **Next Question:** Can we make learning more stable by judging actions relative to "typical" performance?

---

### 2️⃣ Baseline: *How do we reduce the noise?*

**The Insight:**
Absolute rewards don't matter — what matters is whether we did *better or worse than usual*.

**The Fix:**
Subtract a baseline $b(s_t)$ (typically the value function $V^\pi(s_t)$) from the return:

$$
\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a_t|s_t) \, (G_t - b(s_t))] \tag{2}
$$

**Advantage Function:**
Define $A_t = G_t - V^\pi(s_t)$ as the **advantage** — how much better this action was compared to average.

**Mario's Experience:**
If Mario usually scores 500 points but this time scores 800, he knows this run was particularly good. He focuses on *what he did differently*, not the absolute score.

**✅ The Improvement:**
Much lower variance → more stable learning.

**❌ Still a Problem:**
Mario still has to wait until the episode ends to learn anything.

> 🤔 **Next Question:** Can we get feedback *during* the episode instead of waiting until Mario dies?

---

### 3️⃣ Actor-Critic: *Can we learn online, step-by-step?*

**The Insight:**
Instead of waiting for the full episode return $G_t$, estimate it using a **critic** network.

**The Architecture:**
- **Actor** $\pi_\theta(a|s)$: Chooses actions
- **Critic** $V_\phi(s)$: Estimates how good states are

**Temporal-Difference (TD) Advantage:**
Replace $G_t$ with a one-step estimate:

$$
A_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) \tag{3}
$$

This is called the **TD error** — it measures whether the actual reward plus next state's value exceeded our prediction.

**Mario's Experience:**
Every single frame, Mario gets immediate feedback: "Was this jump good or bad?" He doesn't need to die first.

**✅ The Improvement:**
- Online learning (no need to finish episodes)
- Faster, more responsive updates
- Better credit assignment

**❌ Still a Problem:**
Learning is still sample-inefficient (each frame is used once), and training can be unstable.

> 🤔 **Next Question:** Can we collect experience faster and more efficiently?

---

### 4️⃣ A2C/A3C: *What if many Marios learn simultaneously?*

**The Insight:**
Run multiple Mario agents in parallel environments. Each contributes to gradient estimates.

**How it Helps:**
- **Faster data collection**: More experience per second
- **Decorrelated samples**: Different Marios encounter different situations
- **Smoother gradients**: Averaging across agents reduces variance

**✅ The Improvement:**
Training is faster and more stable.

**❌ New Problem:**
Even with these improvements, the policy can still make **sudden, catastrophic changes**.

> 🤔 **Next Question:** How do we prevent Mario from "forgetting" good strategies overnight?

---

### 5️⃣ TRPO: *How do we ensure safe, gradual improvement?*

**The Problem:**
A large policy update can make performance collapse. Imagine Mario suddenly changing from "jump when you see an enemy" to "never jump."

**The Solution:**
Add a **trust region constraint** — restrict how much the new policy $\pi_\theta$ can differ from the old policy $\pi_{\theta_{\text{old}}}$:

$$
\begin{aligned}
\max_\theta \quad & \mathbb{E}_t\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} A_t\right] \\
\text{subject to} \quad & \mathbb{E}_t[\text{KL}(\pi_{\theta_{\text{old}}} \| \pi_\theta)] \le \delta
\end{aligned} \tag{4}
$$

where $\text{KL}(\cdot \| \cdot)$ is the Kullback-Leibler divergence measuring policy difference.

**Mario's Experience:**
Mario takes small, safe steps in learning. He doesn't radically change his jumping strategy overnight.

**✅ The Improvement:**
Much more stable training. Performance rarely degrades.

**❌ New Problem:**
The KL constraint requires **second-order optimization** (computing Hessians), which is computationally expensive.

> 🤔 **Next Question:** Can we keep the stability of TRPO but make it simpler?

---

### 6️⃣ PPO: *Can we simplify safe updates?*

**The Breakthrough:**
Replace the hard KL constraint with a simple **clipped objective** that achieves the same goal.

**The Clipped Surrogate Loss:**

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right] \tag{5}
$$

where the **probability ratio** is:

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \tag{6}
$$

**What the Clip Does:**
- If $r_t(\theta)$ moves outside $[1-\epsilon, 1+\epsilon]$ (typically $\epsilon=0.2$), we stop the gradient
- This prevents large policy changes without expensive second-order computations

**Mario's Experience:**
If Mario's new policy tries to change too much (say, more than ±20%), we gently stop it, keeping updates stable.

**✅ Why PPO Won:**
- Simple to implement (just a few lines of code)
- Computationally efficient (first-order optimization)
- Stable and reliable across diverse tasks
- State-of-the-art performance

**This is the first algorithm implemented in the main repository.**

**❌ The Remaining Challenge:**
PPO learns only from **real interactions** with the environment. Mario must play millions of frames to learn. This is slow, expensive, and sample-inefficient.

> 🤔 **Final Question:** What if Mario could *imagine* playing the game instead of always playing for real?

---

### 7️⃣ DreamerV3: *Can we learn by dreaming?*

**The Paradigm Shift:**
Instead of learning only from real experience, teach Mario to:
1. Build an **internal model** of how the game works
2. **Imagine** future scenarios inside this model
3. Practice and improve his policy inside his imagination
4. Occasionally update the model with real experience

This is called **model-based reinforcement learning**.

---

#### 🧠 How DreamerV3 Works

**Three Core Components:**

**1. World Model**
Learns to simulate the game environment in a compact **latent space**:

- **Encoder** $q_\phi(z_t | o_t)$: Compress observation $o_t$ into latent state $z_t$
- **Dynamics** $p_\phi(z_{t+1} | z_t, a_t)$: Predict next latent state given action
- **Decoder** $p_\phi(o_t | z_t)$: Reconstruct observation from latent state
- **Reward Predictor** $p_\phi(r_t | z_t)$: Predict reward
- **Continue Predictor** $p_\phi(c_t | z_t)$: Predict if episode continues

**2. Imagination Rollouts**
Starting from a real latent state $z_t$, use the world model to generate **imaginary trajectories**:

$$
z_t \xrightarrow{a_t} z_{t+1} \xrightarrow{a_{t+1}} z_{t+2} \xrightarrow{a_{t+2}} \cdots
$$

These trajectories are generated entirely inside the model — no real environment interaction needed.

**3. Actor-Critic in Latent Space**
Train the policy $\pi_\theta(a|z)$ and value function $V_\psi(z)$ on imagined trajectories:

- **Actor objective**: Maximize imagined returns
- **Critic objective**: Accurately predict imagined values

---

#### 🔄 Training Loop

```
1. Collect a batch of real experience from the environment
2. Train the world model to predict observations, rewards, and continuations
3. Sample real states, then generate imaginary rollouts using the world model
4. Train actor and critic on imaginary experience
5. Repeat
```

**Mario's Experience:**
Mario plays the game for a short time, watches what happens, and learns how the world behaves (enemies move, blocks break, jumping has consequences). Then, he **mentally simulates** thousands of different scenarios:
- "What if I jump here?"
- "What if I run past this enemy?"
- "What if I take the pipe?"

He tests all these strategies in his mind before trying them in the real game.

---

#### 🎯 Why DreamerV3 is Powerful

**Benefits:**
- **Sample Efficiency**: Learns from far fewer real environment steps
- **Planning**: Can imagine and evaluate plans before executing them
- **Generalization**: World model captures environment dynamics, enabling better transfer
- **Safety**: Most learning happens in imagination, not in risky real environments

**Trade-offs:**
- **Complexity**: More components to implement and tune
- **Model Error**: If the world model is wrong, the policy may learn suboptimal behaviors
- **Computational Cost**: Training the world model adds overhead

**This is the second algorithm implemented in the main repository.**

---

## 🔁 Summary: The Journey from REINFORCE to DreamerV3

| Algorithm | Key Question | Key Innovation | Mario's Learning Style |
|-----------|--------------|----------------|------------------------|
| **REINFORCE** | Can we learn from rewards? | Policy gradient from episode returns | Pure trial and error |
| **+ Baseline** | How do we reduce noise? | Subtract expected value | Learns relative success |
| **Actor-Critic** | Can we learn online? | TD error for immediate feedback | Real-time learning |
| **A2C/A3C** | Can we learn faster? | Parallel environments | Multiple Marios learn together |
| **TRPO** | How do we prevent collapse? | Trust region constraint | Safe, gradual improvement |
| **PPO** | Can we simplify safety? | Clipped surrogate objective | Efficient and stable |
| **DreamerV3** | Can we learn by imagining? | World model + imagination rollouts | Dreams before acting |

---

## 📘 References and Further Reading

### Original Papers

1. **REINFORCE**
   Williams, R. J. (1992). *Simple statistical gradient-following algorithms for connectionist reinforcement learning.* Machine Learning, 8, 229-256.

2. **Actor-Critic Methods**
   Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

3. **A3C**
   Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., & Kavukcuoglu, K. (2016). *Asynchronous Methods for Deep Reinforcement Learning.* ICML.

4. **TRPO**
   Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). *Trust Region Policy Optimization.* ICML.

5. **PPO**
   Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.

6. **GAE (Generalized Advantage Estimation)**
   Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* arXiv:1506.02438.

7. **DreamerV1**
   Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2019). *Dream to Control: Learning Behaviors by Latent Imagination.* ICLR 2020.

8. **DreamerV2**
   Hafner, D., Lillicrap, T., Norouzi, M., & Ba, J. (2020). *Mastering Atari with Discrete World Models.* ICLR 2021.

9. **DreamerV3**
   Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). *Mastering Diverse Domains through World Models.* arXiv:2301.04104.

### Helpful Resources

- **[Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/)**
  Excellent educational resource with clear explanations and code examples

- **[Deep RL Course (Hugging Face)](https://huggingface.co/deep-rl-course)**
  Free course covering the fundamentals to advanced topics

- **[DreamerV3 Official Implementation (JAX)](https://github.com/danijar/dreamerv3)**
  Official implementation by Danijar Hafner

- **[Sutton & Barto RL Book (Free Online)](http://incompleteideas.net/book/the-book-2nd.html)**
  The canonical textbook on reinforcement learning

- **[PPO Explained (Video by Arxiv Insights)](https://www.youtube.com/watch?v=5P7I-xPq8u8)**
  Visual explanation of PPO

- **[World Models Paper (Ha & Schmidhuber, 2018)](https://worldmodels.github.io/)**
  Precursor to Dreamer with excellent visualizations

- **[OpenAI Baselines](https://github.com/openai/baselines)**
  High-quality implementations of RL algorithms

---

## 💡 Key Takeaways

1. **Start Simple**: REINFORCE shows that policy gradients can work, even if they're noisy
2. **Reduce Variance**: Baselines and critics dramatically improve stability
3. **Learn Online**: TD learning enables step-by-step improvement
4. **Safe Updates**: Trust regions and clipping prevent catastrophic policy changes
5. **Imagine Ahead**: World models enable sample-efficient learning through mental simulation

The evolution of RL algorithms is a story of **identifying problems** and **systematically solving them**. Each algorithm inherits the wisdom of its predecessors while addressing their limitations.

---

**[← Back to Main README](README.md)**
