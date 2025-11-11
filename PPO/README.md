# PPO Algorithm: Usage Guide and Implementation Details

**Paper Reference**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017)

This document provides both a practical usage guide for training and evaluating agents, as well as comprehensive implementation details of the PPO algorithm in this codebase, using Super Mario Bros as a concrete example.

---

## Table of Contents

1. [Quick Start: Usage Guide](#quick-start-usage-guide)
   - [Training the Agent](#1-training-the-agent)
   - [Monitoring Training with TensorBoard](#2-monitoring-training-with-tensorboard)
   - [Playing Mario with the Trained Model](#3-playing-mario-with-the-trained-model)
2. [Key Concepts and Notation](#1-key-concepts-and-notation)
3. [Major Components](#2-major-components)
4. [Loss Functions](#3-loss-functions)
5. [Training Process](#4-training-process)
6. [Implementation Tricks](#5-implementation-tricks)

---

## Quick Start: Usage Guide

### 1. Training the Agent

**Basic Training (Standard Configuration):**
```bash
# Train with default hyperparameters
python train_mario_ppo.py --configs defaults
```

**Exploration Mode (When Stuck in Local Optima):**

If your agent gets stuck in suboptimal behaviors (e.g., always dying at the same spot):
```bash
# Increase entropy bonus for more exploration
python train_mario_ppo.py --configs defaults explore
```

**What `explore` does:**
- **Higher Entropy Bonus**: Increases ent_coef from 0.03→0.05 to encourage trying diverse actions
- **Lower Learning Rate**: Reduces learning rate to 0.0001 for more stable exploration
- **Use When**: Agent is stuck in local optima or not discovering new strategies

**Other Training Options:**

```bash
# Fast mode (for quicker iteration during development)
python train_mario_ppo.py --configs defaults fast

# Debug mode (very fast testing of code changes)
python train_mario_ppo.py --configs debug

# Resume from checkpoint
python train_mario_ppo.py --resume --logdir ./logdir/mario_ppo

# Custom log directory
python train_mario_ppo.py --configs defaults --logdir ./my_experiments/ppo_run1
```

**Training Progress:**
- The agent will train for 10,000,000 timesteps (or 100,000 in debug mode)
- Evaluation runs every 100,000 timesteps
- Checkpoints are saved every 100,000 timesteps in `logdir/mario_ppo/`
- Training takes ~8-12 hours on RTX 3080 GPU

**Key Differences from DreamerV3:**
- PPO uses **32 parallel environments** for efficient on-policy data collection
- Collects 128 steps × 32 envs = 4,096 transitions per update
- Requires more environment timesteps but faster wall-clock time per update
- More straightforward algorithm but less sample efficient

---

### 2. Monitoring Training with TensorBoard

PPO logs extensive metrics and visualizations to TensorBoard. Launch it to monitor training in real-time:

```bash
# Start TensorBoard (point it to your log directory)
tensorboard --logdir ./logdir/mario_ppo

# Then open http://localhost:6006 in your browser
```

**Key Metrics to Monitor:**

**Scalars Tab:**
- `eval/mean_return`: Average episode return during evaluation (target: ~2500-3500 for completion)
- `eval/success_rate`: Fraction of episodes that reached the flag (target: >80%)
- `train/policy_loss`: PPO clipped objective loss
- `train/value_loss`: Critic MSE loss (should decrease)
- `train/entropy`: Policy entropy (should stay > 0 for exploration)
- `train/approx_kl`: KL divergence between old and new policy (should be small, < 0.05)
- `train/clip_fraction`: Fraction of ratios being clipped (typical: 0.1-0.3)
- `train/explained_variance`: How well value function predicts returns (target: >0.7)

**Understanding Key Metrics:**

**Policy Loss (PPO Objective):**
- Negative of clipped surrogate objective
- Should decrease initially, then stabilize
- Large fluctuations indicate instability (reduce learning rate)

**Entropy:**
- Measures policy randomness/exploration
- Should start high (~1.5-2.0) and gradually decrease
- If drops too fast (<0.1), increase `ent_coef`
- If stays too high (>1.5), agent isn't learning a clear strategy

**Clip Fraction:**
- Shows how often policy updates are being clipped
- Too high (>0.5): policy changing too fast, reduce learning rate
- Too low (<0.05): clipping not active, could increase learning rate

**Explained Variance:**
- How well critic predicts actual returns
- High (>0.8): critic is accurate, good advantage estimates
- Low (<0.5): critic is poor, advantages are noisy

**Images Tab (Evaluation Videos):**

PPO records evaluation episodes as videos, showing the agent's behavior:

<p align="center">
  <img src="../assets/episode_002_reward_4093_steps_283.gif" alt="PPO agent playing Mario" width="400"/>
</p>

**What to Look For:**
- **Early Training**: Random movements, frequent deaths
- **Mid Training**: Learns to move right, avoid simple pits
- **Late Training**: Consistent flag completion, optimal paths
- **Success Indicator**: Agent consistently reaches flag in <300 steps

The videos show up to 3 episodes side-by-side, helping you spot patterns in behavior.

---

### 3. Playing Mario with the Trained Model

Once training is complete (or even during training), you can watch your agent play:

**Basic Usage:**
```bash
# Play 5 episodes using the best checkpoint (deterministic policy by default)
python play_mario_ppo.py --logdir logdir/mario_ppo --episodes 5
```

**Stochastic Mode (Exploration):**
```bash
# Use stochastic policy (sample actions instead of taking most likely)
python play_mario_ppo.py --logdir logdir/mario_ppo --episodes 5 --stochastic
```

**Use Latest Checkpoint:**
```bash
# Play with the most recent checkpoint instead of best
python play_mario_ppo.py --logdir logdir/mario_ppo --checkpoint latest.pt
```

**Save Episodes as GIFs:**
```bash
# Record episodes and save as animated GIFs
python play_mario_ppo.py --logdir logdir/mario_ppo --episodes 5 --save-gif

# Customize GIF settings
python play_mario_ppo.py --logdir logdir/mario_ppo --episodes 5 --save-gif --gif-fps 30 --gif-dir ./my_gifs
```

**Options:**
- `--logdir`: Directory containing the trained model checkpoint
- `--episodes`: Number of episodes to play (default: 5)
- `--checkpoint`: Checkpoint file to load ('best.pt' or 'latest.pt', default: 'best.pt')
- `--stochastic`: Use stochastic policy instead of deterministic (default: deterministic)
- `--save-gif`: Save each episode as an animated GIF
- `--gif-fps`: Frames per second for saved GIFs (default: 20)
- `--gif-dir`: Directory to save GIF files (default: logdir/gifs)

**What to Expect:**
- The agent should reach the flag after ~3-5M timesteps
- Final reward should be around 2500-3500 (includes flag bonus of 1000)
- Success rate typically 60-80% after full training
- Episode length typically 250-350 steps with action_repeat=4

**Comparing to DreamerV3:**
- PPO: Needs more timesteps (10M vs 400K) but trains faster in wall-clock time
- PPO: More direct policy learning, easier to interpret
- DreamerV3: More sample efficient, better for expensive environments

---

## 1. Key Concepts and Notation

PPO is an **on-policy** policy gradient method that learns directly from experience. It uses a clipped objective to prevent destructively large policy updates.

### 1.1 Observations and Actions

| Symbol | Dimension | Description | Mario Example |
|--------|-----------|-------------|---------------|
| $s_t$ | (64, 64, 3) | State/observation (RGB image) at time t | Screenshot of Mario game (64×64 RGB) showing Mario, enemies, blocks |
| $a_t$ | (7,) | Action taken at time t (one-hot) | One of 7 discrete actions: NOOP, right, right+A, right+B, right+A+B, A (jump), left |
| $r_t$ | (1,) | Reward received at time t | Distance traveled + 1000 for flag - penalties for dying |
| $\text{done}_t$ | (1,) | Episode termination flag | 1 when Mario dies or reaches flag, 0 otherwise |

### 1.2 Policy and Value Function

| Symbol | Type | Description | Purpose |
|--------|------|-------------|---------|
| $\pi_\theta(a\|s)$ | Policy | Probability distribution over actions | Agent's policy parameterized by $\theta$ |
| $V_\phi(s)$ | Value Function | Expected cumulative return from state s | Baseline for advantage estimation, parameterized by $\phi$ |
| $A(s,a)$ | Advantage | How much better action a is than average | $A(s,a) = Q(s,a) - V(s)$ |

### 1.3 PPO-Specific Terms

| Symbol | Description | Purpose |
|--------|-------------|---------|
| $\pi_{old}(a\|s)$ | Old policy (before update) | Reference for computing probability ratio |
| $r_t = \frac{\pi_\theta(a\|s)}{\pi_{old}(a\|s)}$ | Probability ratio | Measures how much policy changed |
| $\varepsilon$ | Clipping parameter (0.2) | Limits policy update magnitude |
| $\gamma$ | Discount factor (0.99) | Temporal discounting |
| $\lambda$ | GAE parameter (0.95) | Bias-variance tradeoff |

### 1.4 Time Convention

Throughout this document:
- **t** denotes the current timestep
- **t+1** denotes the next timestep

**Example sequence in Mario:**
1. At $t=0$: Mario is standing, observes initial screen → $s_0$
2. Policy selects action: $a_0 = \text{"right"} \sim \pi_\theta(\cdot|s_0)$
3. Environment steps: $s_1, r_1 = \text{env.step}(a_0)$
4. Reward: $r_1 = +1$ for distance traveled
5. Continue until $\text{done}_t = 1$

---

## 2. Major Components

PPO consists of three main components: the policy network (actor), the value network (critic), and a rollout buffer for storing on-policy experience.

### 2.1 Actor (Policy Network)

**Mathematical Definition:**

$$\pi_\theta(a|s) = \text{Categorical}(\text{logits}_\theta(s))$$

**Purpose:** Maps states to action probabilities.

**Architecture:** CNN + MLP

```
Input: s_t ∈ R^(64×64×3)
  ↓
Conv2d(32, 8×8, stride=4) + ReLU  →  (16×16×32)
  ↓
Conv2d(64, 4×4, stride=2) + ReLU  →  (8×8×64)
  ↓
Conv2d(64, 3×3, stride=1) + ReLU  →  (8×8×64)
  ↓
Flatten  →  features ∈ R^4096
  ↓
Linear(512) + ReLU  →  hidden ∈ R^512
  ↓
Linear(7)  →  logits ∈ R^7
  ↓
Softmax  →  π_θ(a|s_t)
```

**Implementation:**
- Class: [`ActorCriticNetwork`](networks.py#L35) in [networks.py](networks.py)
- Initialization: Orthogonal weights for CNN, small initialization (0.01) for policy head
- Activation: ReLU throughout

**Mario Example:**

For a given state $s_t$, outputs action probabilities:

```
π_θ(a|s_t) = [0.05, 0.70, 0.15, 0.03, 0.03, 0.02, 0.02]
             [NOOP, right, right+A, right+B, right+A+B, A, left]
```

Most likely action: "right" (keep moving forward).

---

### 2.2 Critic (Value Network)

**Mathematical Definition:**

$$V_\phi(s) = \text{MLP}_\phi(\text{CNN}(s))$$

**Purpose:** Estimates expected cumulative return from state s, used for computing advantages.

**Architecture:** Shared CNN + separate value head

```
Input: s_t ∈ R^(64×64×3)
  ↓
Shared CNN (same as actor)  →  features ∈ R^4096
  ↓
Linear(512) + ReLU  →  hidden ∈ R^512
  ↓
Linear(1)  →  V_φ(s_t) ∈ R
```

**Implementation:**
- Class: [`ActorCriticNetwork`](networks.py#L35) in [networks.py](networks.py)
- Shares CNN features with actor for efficiency
- Standard initialization (1.0) for value head

**Mario Example:**
- Mario near flag: $V(s) \approx 1200$ (high value, about to win)
- Mario about to fall in pit: $V(s) \approx -50$ (low value, about to die)
- Mario in middle of level: $V(s) \approx 500$ (moderate value)

---

### 2.3 Rollout Buffer

**Purpose:** Stores on-policy experience from parallel environments and computes Generalized Advantage Estimation (GAE).

**What it Stores:**
- States: (num_steps, num_envs, C, H, W)
- Actions: (num_steps, num_envs)
- Rewards: (num_steps, num_envs)
- Values: (num_steps, num_envs)
- Log probabilities: (num_steps, num_envs)
- Dones: (num_steps, num_envs)

**GAE Computation:**

Generalized Advantage Estimation (Schulman et al., 2016) computes advantages using exponentially-weighted average of TD residuals:

$$\delta_t = r_t + \gamma \cdot V(s_{t+1}) \cdot (1 - \text{done}_t) - V(s_t)$$

$$\hat{A}_t = \delta_t + (\gamma\lambda) \cdot \delta_{t+1} + (\gamma\lambda)^2 \cdot \delta_{t+2} + \ldots$$

**Parameters:**
- $\gamma = 0.99$: Discount factor for future rewards
- $\lambda = 0.95$: GAE parameter (bias-variance tradeoff)

**Implementation:**
- Class: [`RolloutBuffer`](rollout_buffer.py#L13) in [rollout_buffer.py](rollout_buffer.py)
- Computes advantages backward in time
- Normalizes advantages per batch for stability

**Mario Example:**

t=0: r=1, $V(s_0)=100$, $V(s_1)=110$ → $\delta_0 = 1 + 0.99 \cdot 110 - 100 = 9.9$
t=1: r=1, $V(s_1)=110$, $V(s_2)=120$ → $\delta_1 = 1 + 0.99 \cdot 120 - 110 = 9.8$
t=2: r=1000, done=1 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ $\delta_2 = 1000 + 0 - 120 = 880$

$\hat{A}_0 = 9.9 + 0.95 \cdot 0.99 \cdot 9.8 + (0.95 \cdot 0.99)^2 \cdot 880 \approx 800$ (propagated flag reward)

---

### 2.4 Vectorized Environments

**Purpose:** Run multiple Mario environments in parallel for efficient on-policy data collection.

**Architecture:** SubprocVecEnv
- Launches 32 separate Python processes
- Each process runs independent Mario environment
- True parallelism via multiprocessing
- Auto-resets environments when episodes end

**Benefits:**
- 32x faster data collection than single environment
- Diverse experience from different random seeds
- Essential for efficient on-policy learning (PPO needs fresh data)

**Implementation:**
- Class: [`SubprocVecEnv`](../envs/vec_mario.py#L72) in [vec_mario.py](../envs/vec_mario.py)
- Helper: [`make_vec_mario_env()`](../envs/vec_mario.py#L184)
- Each process communicates via pipes

**Mario Example:**
```python
# Create 32 parallel environments
envs = make_vec_mario_env(num_envs=32, seed=0)

# Step all environments at once
actions = agent.get_actions(obs)  # (32,)
obs, rewards, dones, infos = envs.step(actions)
# obs: (32, 64, 64, 3) - batch of observations
# rewards: (32,) - batch of rewards
# dones: (32,) - batch of done flags
```

**Why PPO Needs This:**
- On-policy: Can't reuse old data like DreamerV3
- Needs 4,096+ transitions per update
- Single environment too slow (128 steps = 2-3 minutes)
- 32 environments: 128 steps = 4-5 seconds

---

## 3. Loss Functions

PPO training involves three loss components: the clipped surrogate objective (policy loss), value function loss (critic loss), and entropy bonus.

### 3.1 Clipped Surrogate Objective (Policy Loss)

**Mathematical Definition:**

PPO's key innovation is the clipped objective that prevents destructively large policy updates:

$$L^{CLIP}(\theta) = \mathbb{E}_t [\min(r_t(\theta) \cdot \hat{A}_t, \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \cdot \hat{A}_t)]$$

where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$ is the probability ratio
- $\hat{A}_t$ is the advantage estimate
- $\varepsilon = 0.2$ is the clipping parameter

**Clipping Function:**

$$\text{clip}(r, 1-\varepsilon, 1+\varepsilon) = \max(1-\varepsilon, \min(r, 1+\varepsilon)) = \begin{cases}
1-\varepsilon & \text{if } r < 1-\varepsilon \text{ (policy decreased too much)} \\
r & \text{if } 1-\varepsilon \leq r \leq 1+\varepsilon \text{ (acceptable change)} \\
1+\varepsilon & \text{if } r > 1+\varepsilon \text{ (policy increased too much)}
\end{cases}$$

**Purpose:**
- Prevents large policy updates that could destabilize training
- Ensures $\pi_\theta$ stays close to $\pi_{old}$ (old policy from data collection)
- Conservative policy improvement

**How it Works:**

The clipping creates a "trust region" around the old policy:

1. **If advantage is positive** ($\hat{A}_t > 0$): Good action, want to increase probability
   - If $r > 1+\varepsilon$: Already increased too much → clip to $1+\varepsilon$
   - If $r \leq 1+\varepsilon$: Normal update

2. **If advantage is negative** ($\hat{A}_t < 0$): Bad action, want to decrease probability
   - If $r < 1-\varepsilon$: Already decreased too much → clip to $1-\varepsilon$
   - If $r \geq 1-\varepsilon$: Normal update

**Implementation:**
```python
# Compute probability ratio: r_t = π_θ / π_old
ratio = torch.exp(log_probs - old_log_probs)

# Compute clipped objective
advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
policy_loss1 = advantage * ratio
policy_loss2 = advantage * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
```

**Code Location:** [`PPOAgent.update()`](ppo_agent.py#L144) in [ppo_agent.py](ppo_agent.py)

**Mario Example:**

Action: "right+A" (jump forward)
Old policy: $\pi_{old}(\text{right+A}|s) = 0.2$
New policy: $\pi_\theta(\text{right+A}|s) = 0.3$
Ratio: $r = 0.3 / 0.2 = 1.5$

Advantage: $\hat{A} = +50$ (good action, led to progress)

Unclipped: $1.5 \times 50 = 75$
Clipped: $\text{clip}(1.5, 0.8, 1.2) \times 50 = 1.2 \times 50 = 60$

Final: $\min(75, 60) = 60$ &nbsp;&nbsp;[clipping prevented overly aggressive update]

---

### 3.2 Value Function Loss (Critic Loss)

**Mathematical Definition:**

$$L^{VF}(\phi) = \mathbb{E}_t [(V_\phi(s_t) - V^{target}_t)^2]$$

where $V^{target}_t = \hat{A}_t + V_{old}(s_t)$ (advantage + old value = return estimate)

**Purpose:** Trains the critic to accurately predict returns, which is essential for computing good advantage estimates.

**Why Not Direct Return?**
- Returns have high variance
- Value target ($\hat{A} + V_{old}$) is lower variance due to GAE
- Better gradient estimates

**Implementation:**
```python
# Compute value loss
value_pred = critic(states)
value_target = advantages + old_values  # Returns estimate
value_loss = F.mse_loss(value_pred, value_target.detach())
```

**Code Location:** [`PPOAgent.update()`](ppo_agent.py#L161) in [ppo_agent.py](ppo_agent.py)

**Mario Example:**

State: Mario standing before pit
Old value: $V_{old}(s) = 100$
Advantage: $\hat{A} = -50$ (agent died after this state)
Target: $V^{target} = -50 + 100 = 50$

Predicted: $V_\phi(s) = 100$ (critic too optimistic)
Loss: $(100 - 50)^2 = 2500$ (large error)

After update: $V_\phi(s) \to 60$ (closer to target)

---

### 3.3 Entropy Bonus

**Mathematical Definition:**

$$H[\pi_\theta] = \mathbb{E}_s \left[-\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)\right]$$

**Purpose:** Encourages exploration by preventing policy from becoming too deterministic too early.

**Effect:**
- Higher entropy → more uniform action distribution → more exploration
- Lower entropy → peaky action distribution → more exploitation

**Coefficient:** $c_2 = 0.03$ (ent_coef)
- Higher values (0.05): More exploration
- Lower values (0.01): More exploitation

**Implementation:**
```python
# Compute entropy bonus
dist = Categorical(logits=logits)
entropy = dist.entropy().mean()

# Add to loss (negative because we maximize entropy)
total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
```

**Code Location:** [`PPOAgent.update()`](ppo_agent.py#L168) in [ppo_agent.py](ppo_agent.py)

**Mario Example:**

High entropy policy (early training):
$\pi_\theta$ = [0.15, 0.17, 0.14, 0.13, 0.16, 0.12, 0.13] &nbsp;&nbsp;$H \approx 1.95$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[fairly uniform → explores different actions]

Low entropy policy (late training):
$\pi_\theta$ = [0.02, 0.85, 0.08, 0.02, 0.01, 0.01, 0.01] &nbsp;&nbsp;$H \approx 0.60$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[peaked at "right" → exploits known good action]

---

### 3.4 Total Loss

**Combined Objective:**

$$L(\theta, \phi) = -L^{CLIP}(\theta) + c_1 \cdot L^{VF}(\phi) - c_2 \cdot H[\pi_\theta]$$

where:
- $c_1 = 0.5$ is the value loss coefficient (vf_coef)
- $c_2 = 0.03$ is the entropy coefficient (ent_coef)

**Interpretation:**
- Maximize clipped objective (hence negative sign)
- Minimize value error
- Maximize entropy (hence negative sign)

**Trade-offs:**
- $c_1$ too high: Over-focus on value, neglect policy
- $c_1$ too low: Poor value estimates, noisy advantages
- $c_2$ too high: Too much exploration, slow learning
- $c_2$ too low: Premature convergence, local optima

**Implementation:** [`PPOAgent.update()`](ppo_agent.py#L170) in [ppo_agent.py](ppo_agent.py)

---

## 4. Training Process

This section describes the PPO training loop and how it differs from DreamerV3.

### 4.1 High-Level Algorithm Overview

PPO follows an on-policy actor-critic loop:

1. **Collect Rollouts (On-Policy Data)**
   - Run policy $\pi_{old}$ in 32 parallel environments
   - Collect 128 steps per environment = 4,096 transitions
   - Store: states, actions, rewards, values, log_probs, dones

2. **Compute Advantages**
   - Use Generalized Advantage Estimation (GAE)
   - Compute advantage $\hat{A}_t$ for each timestep
   - Normalize advantages per batch

3. **Update Policy and Value Function**
   - For K=4 epochs:
     - Shuffle data into mini-batches of size 256
     - For each mini-batch:
       - Compute probability ratio $r_t$
       - Compute clipped objective
       - Compute value loss
       - Compute entropy
       - Backpropagate total loss
       - Clip gradients

4. **Discard Data and Repeat**
   - Data is on-policy, discard after update
   - Collect new rollouts with updated policy
   - Repeat until 10M timesteps

**Key Insight:** PPO is simple and stable due to clipped objective. It doesn't need complex world models or replay buffers, just fresh on-policy data.

---

### 4.2 Implementation: Training Loop Pseudo-Code

Below is the actual training structure based on the codebase:

```python
# ============================================================================
# INITIALIZATION (train_mario_ppo.py)
# ============================================================================

# 1. Setup vectorized environments
envs = make_vec_mario_env(num_envs=32, seed=config.seed)

# 2. Create PPO agent
agent = PPOAgent(
    obs_space=envs.observation_space,
    action_space=envs.action_space,
    feature_dim=512,
    learning_rate=0.00025,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    vf_coef=0.5,
    ent_coef=0.03,
    max_grad_norm=0.5,
    device=device
)
#    ↓ Creates:
#    - actor_critic: ActorCriticNetwork  # networks.py:35
#    - optimizer: Adam optimizer
#    - rollout_buffer: RolloutBuffer     # rollout_buffer.py:13

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

obs = envs.reset()  # (32, 64, 64, 3)
done = np.zeros(32)
global_step = 0

while global_step < 10_000_000:  # 10M timesteps

    # ========================================================================
    # STEP 1: COLLECT ROLLOUT (128 steps × 32 envs = 4096 transitions)
    # ========================================================================

    for step in range(128):
        global_step += 32  # 32 environments step in parallel

        # (a) Select actions with old policy π_old
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(device)

            # Get action distribution from actor
            logits = actor_critic.get_action_logits(obs_tensor)
            dist = Categorical(logits=logits)

            # Sample actions
            actions = dist.sample()  # shape: (32,)
            log_probs = dist.log_prob(actions)  # shape: (32,)

            # Get value estimates
            values = actor_critic.get_values(obs_tensor)  # shape: (32,)

        # (b) Step environments
        next_obs, rewards, dones, infos = envs.step(actions.cpu().numpy())
        # next_obs: (32, 64, 64, 3)
        # rewards: (32,)
        # dones: (32,)

        # (c) Store in rollout buffer
        rollout_buffer.add(
            obs=obs,
            action=actions,
            reward=rewards,
            done=dones,
            value=values,
            log_prob=log_probs
        )

        obs = next_obs

    # ========================================================================
    # STEP 2: COMPUTE ADVANTAGES
    # ========================================================================

    # Get value for last observation (for bootstrapping)
    with torch.no_grad():
        last_values = actor_critic.get_values(torch.FloatTensor(obs).to(device))

    # Compute advantages using GAE
    # Formula: Â_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
    # where: δ_t = r_t + γ·V(s_{t+1})·(1-done_t) - V(s_t)
    advantages, returns = rollout_buffer.compute_advantages_and_returns(
        last_values=last_values,
        gamma=0.99,
        gae_lambda=0.95
    )

    # ========================================================================
    # STEP 3: UPDATE POLICY AND VALUE FUNCTION
    # ========================================================================

    # Flatten batch: (128 steps, 32 envs) → (4096 transitions)
    batch = rollout_buffer.get_batch()

    # Update for K=4 epochs
    for epoch in range(4):

        # Shuffle and create mini-batches of size 256
        # 4096 / 256 = 16 mini-batches per epoch
        indices = np.random.permutation(4096)

        for start in range(0, 4096, 256):
            end = start + 256
            mb_indices = indices[start:end]

            # Get mini-batch data
            mb_obs = batch['obs'][mb_indices]
            mb_actions = batch['actions'][mb_indices]
            mb_old_log_probs = batch['log_probs'][mb_indices]
            mb_advantages = batch['advantages'][mb_indices]
            mb_returns = batch['returns'][mb_indices]

            # (a) Forward pass with current policy π_θ
            logits = actor_critic.get_action_logits(mb_obs)
            dist = Categorical(logits=logits)

            # Get new log probs
            new_log_probs = dist.log_prob(mb_actions)

            # Get entropy
            entropy = dist.entropy().mean()

            # Get new values
            new_values = actor_critic.get_values(mb_obs).squeeze()

            # (b) Compute probability ratio
            ratio = torch.exp(new_log_probs - mb_old_log_probs)

            # (c) Normalize advantages
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # (d) Compute clipped objective
            policy_loss1 = mb_advantages * ratio
            policy_loss2 = mb_advantages * torch.clamp(ratio, 1-0.2, 1+0.2)
            policy_loss = -torch.min(policy_loss1, policy_loss2).mean()

            # (e) Compute value loss
            value_loss = F.mse_loss(new_values, mb_returns)

            # (f) Compute total loss
            loss = policy_loss + 0.5 * value_loss - 0.03 * entropy

            # (g) Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5)
            optimizer.step()

    # Clear rollout buffer (data is on-policy, can't reuse)
    rollout_buffer.reset()

    # ========================================================================
    # STEP 4: EVALUATION AND LOGGING
    # ========================================================================

    if global_step % 100_000 == 0:
        # Evaluate for 10 episodes
        eval_stats = evaluate_agent(agent, eval_env, num_episodes=10)

        # Log metrics
        print(f"Step {global_step}")
        print(f"  Mean Return: {eval_stats['mean_return']:.1f}")
        print(f"  Success Rate: {eval_stats['success_rate']:.1%}")

        # Log to TensorBoard
        writer.add_scalar('eval/mean_return', eval_stats['mean_return'], global_step)
        writer.add_scalar('eval/success_rate', eval_stats['success_rate'], global_step)

        # Log training metrics
        writer.add_scalar('train/policy_loss', policy_loss.item(), global_step)
        writer.add_scalar('train/value_loss', value_loss.item(), global_step)
        writer.add_scalar('train/entropy', entropy.item(), global_step)

    if global_step % 100_000 == 0:
        # Save checkpoint
        torch.save({
            'step': global_step,
            'model_state_dict': actor_critic.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, logdir / f'checkpoint_{global_step}.pt')
```

---

### 4.3 Key Implementation Details

**Gradient Flow:**
- Policy loss: Gradients flow through actor
- Value loss: Gradients flow through critic
- Both share CNN features (efficient feature learning)

**Efficiency:**
- 32 parallel environments: ~4-5 seconds per rollout
- 4 epochs × 16 mini-batches = 64 updates per rollout
- Total: ~10-15 seconds per update cycle
- Much faster than DreamerV3 per update, but needs more data

**Memory Management:**
- Rollout buffer: 128 × 32 = 4,096 transitions
- Mini-batch: 256 transitions
- GPU memory: ~2-3 GB for model + data
- Much lower memory than DreamerV3's replay buffer

**On-Policy Constraint:**
- Data discarded after each update
- Must collect fresh data with current policy
- No replay buffer (unlike DreamerV3)
- This is why we need 32 parallel environments

---

## 5. Implementation Tricks

PPO uses several important techniques for stable and efficient training.

---

### 5.1 Advantage Normalization

**Purpose:** Standardizes advantages to have mean 0 and std 1, improving optimization stability.

**Formula:**

$$\hat{A}_{\text{normalized}} = \frac{\hat{A} - \text{mean}(\hat{A})}{\text{std}(\hat{A}) + 10^{-8}}$$

**Why it works:**
- Prevents advantages from having very different scales
- Makes learning rate more consistent across batches
- Reduces sensitivity to reward scaling

**Implementation:**
```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**Code Location:** [`PPOAgent.update()`](ppo_agent.py#L153) in [ppo_agent.py](ppo_agent.py)

---

### 5.2 Gradient Clipping

**Purpose:** Prevents exploding gradients that can destabilize training.

**Formula:**
```
if ||g|| > max_norm:
    g ← g * max_norm / ||g||
```

**Parameters:**
- max_norm = 0.5 (clip gradient norm to 0.5)

**Implementation:**
```python
torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_norm=0.5)
```

**Code Location:** [`PPOAgent.update()`](ppo_agent.py#L172) in [ppo_agent.py](ppo_agent.py)

**Why it works:**
- Limits maximum step size in parameter space
- Prevents sudden parameter changes from outlier gradients
- Essential for stable RL training

---

### 5.3 Orthogonal Initialization

**Purpose:** Initializes weights to preserve gradient flow through deep networks.

**Formula:**
```
W ← Q from QR decomposition of random matrix
scale by gain factor
```

**Gains:**
- CNN layers: gain = √2 (for ReLU)
- Policy head: gain = 0.01 (small init)
- Value head: gain = 1.0 (standard init)

**Implementation:**
```python
def init_weights(module, gain=np.sqrt(2)):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
```

**Code Location:** [`ActorCriticNetwork._init_weights()`](networks.py#L90) in [networks.py](networks.py)

**Why it works:**
- Orthogonal matrices preserve norm of vectors
- Prevents gradient vanishing/exploding in deep networks
- Small policy head init prevents overly confident initial policy

---

### 5.4 Value Function Clipping (Optional, Not Used)

**Note:** PPO sometimes uses value clipping, but we don't enable it by default as it can harm performance:

```python
# Optional: clip value function similar to policy
value_pred_clipped = old_values + torch.clamp(
    value_pred - old_values,
    -clip_coef,
    clip_coef
)
```

**Why we don't use it:**
- Can prevent critic from learning large value changes
- Not necessary when value loss coefficient is properly tuned
- Original PPO paper found mixed results

---

### 5.5 Early Stopping (via KL Divergence Monitoring)

**Purpose:** Stop updates if policy changes too much from old policy.

**Formula:**

$$\text{KL}(\pi_{old} || \pi_\theta) \approx \mathbb{E}[\log \pi_{old}(a|s) - \log \pi_\theta(a|s)] = \mathbb{E}[\text{old\_log\_probs} - \text{new\_log\_probs}]$$

**Threshold:** Typically stop if KL > 0.01-0.015

**Implementation (monitoring only):**
```python
with torch.no_grad():
    approx_kl = (old_log_probs - new_log_probs).mean().item()
    if approx_kl > 0.015:
        print(f"Warning: High KL divergence: {approx_kl:.4f}")
```

**Code Location:** [`PPOAgent.update()`](ppo_agent.py#L157) in [ppo_agent.py](ppo_agent.py)

**Why it works:**
- Prevents policy from changing too rapidly
- Maintains on-policy assumption (data collected with $\pi_{old}$)
- Complements clipping for conservative updates

---

### 5.6 Explained Variance

**Purpose:** Diagnostic metric showing how well value function predicts returns.

**Formula:**

$$\text{EV} = 1 - \frac{\text{Var}(\text{returns} - \text{values})}{\text{Var}(\text{returns})}$$

**Interpretation:**
- $\text{EV} \approx 1.0$: Perfect predictions
- $\text{EV} \approx 0.0$: No better than predicting mean
- $\text{EV} < 0.0$: Worse than predicting mean (bad critic)

**Target:** Should be > 0.7 for good training

**Implementation:**
```python
y_pred = values
y_true = returns
var_y = torch.var(y_true)
explained_var = 1 - torch.var(y_true - y_pred) / var_y
```

**Why it matters:**
- Low EV means advantages are noisy
- High EV means good advantage estimates
- Useful for debugging value function issues

---

### Summary of Implementation Tricks

| Trick | Purpose | Key Parameters | Implementation |
|-------|---------|----------------|----------------|
| **Clipped Objective** | Prevent large policy updates | ε=0.2 | [ppo_agent.py:150](ppo_agent.py#L150) |
| **Advantage Normalization** | Stabilize learning | per-batch | [ppo_agent.py:153](ppo_agent.py#L153) |
| **Gradient Clipping** | Prevent exploding gradients | max_norm=0.5 | [ppo_agent.py:172](ppo_agent.py#L172) |
| **Orthogonal Init** | Better gradient flow | gains=[√2, 0.01, 1.0] | [networks.py:90](networks.py#L90) |
| **GAE** | Reduce advantage variance | λ=0.95 | [rollout_buffer.py:57](rollout_buffer.py#L57) |
| **Entropy Bonus** | Encourage exploration | coef=0.03 | [ppo_agent.py:168](ppo_agent.py#L168) |
| **Vectorized Envs** | Efficient data collection | n=32 | [vec_mario.py:184](../envs/vec_mario.py#L184) |
| **Multiple Epochs** | Better sample efficiency | K=4 | [ppo_agent.py:125](ppo_agent.py#L125) |

---

## References

1. **PPO Paper**: Schulman et al., 2017. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
2. **GAE Paper**: Schulman et al., 2016. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
3. **OpenAI Spinning Up**: [PPO Documentation](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
4. **CleanRL Implementation**: [ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)

---

**Last Updated:** 2024

For questions about this implementation, please refer to the papers or explore the linked code sections.
