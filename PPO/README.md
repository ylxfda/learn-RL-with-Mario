# PPO Algorithm: Usage Guide and Implementation Details

**Paper Reference**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017)

This document provides both a practical usage guide for training and evaluating the PPO model, as well as comprehensive implementation details in this codebase.

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

PPO records evaluation episodes as videos, showing the agent's behavior.

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

**Action Button Explanation:**
- **A button**: Jump button (NES controller)
- **B button**: Run/sprint button (also shoots fireballs when powered up)
- **right**: D-pad direction to move right
- **left**: D-pad direction to move left
- **right+A**: Move right while jumping (simultaneous button press)
- **right+B**: Run right (simultaneous button press)
- **right+A+B**: Run right while jumping (all three pressed simultaneously)
- **NOOP**: No operation (no buttons pressed)

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
Input: s_t ∈ R^(64×64×3) # Game scence
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

**Equation 1: TD Residual (Temporal Difference Error)**

$$\delta_t = r_t + \gamma \cdot V(s_{t+1}) \cdot (1 - \text{done}_t) - V(s_t)$$

**What this means:**
- $\delta_t$ measures the **one-step prediction error** of the value function
- $r_t + \gamma \cdot V(s_{t+1})$: **TD target** = immediate reward + discounted future value
- $V(s_t)$: **Current prediction** = what the critic thinks this state is worth
- $(1 - \text{done}_t)$: **Terminal state handling** = 0 if episode ends (no future value), 1 otherwise

**Intuition:**
- If $\delta_t > 0$: The state was **better than expected** (positive surprise)
- If $\delta_t < 0$: The state was **worse than expected** (negative surprise)
- If $\delta_t \approx 0$: The value function is **accurate** (no surprise)

**Why multiply by $(1 - \text{done}_t)$?**
- When episode ends ($\text{done}_t = 1$), there is no next state
- Future value should be 0: $V(s_{t+1}) \cdot (1-1) = 0$
- Only the immediate reward $r_t$ matters for terminal states

---

**Equation 2: GAE Advantage Estimate**

$$\hat{A}_t = \delta_t + (\gamma\lambda) \cdot \delta_{t+1} + (\gamma\lambda)^2 \cdot \delta_{t+2} + \ldots$$

**What this means:**
- $\hat{A}_t$ is a **weighted sum of all future TD errors** from time $t$ onwards
- Each future error $\delta_{t+k}$ is weighted by $(\gamma\lambda)^k$
- Errors further in the future get exponentially smaller weights

**Intuition:**
- **Don't just look at one step**: A single $\delta_t$ can be noisy
- **Look at multiple steps**: Average many TD errors to reduce variance
- **Weight nearby steps more**: Recent errors are more relevant than distant ones

**Why this exponential weighting?**
- $\lambda = 1$: Use **all future errors** equally → low bias, high variance (like Monte Carlo)
- $\lambda = 0$: Use **only current error** → high bias, low variance (like 1-step TD)
- $\lambda = 0.95$: **Balanced tradeoff** → moderate bias and variance

**Expanded form:**
$$\hat{A}_t = \delta_t + (\gamma\lambda) \cdot \delta_{t+1} + (\gamma\lambda)^2 \cdot \delta_{t+2} + (\gamma\lambda)^3 \cdot \delta_{t+3} + \ldots$$

**Weights decay example** (with $\gamma=0.99$, $\lambda=0.95$):
- Step $t$: weight = $1.0$ (100%)
- Step $t+1$: weight = $0.99 \times 0.95 = 0.94$ (94%)
- Step $t+2$: weight = $(0.99 \times 0.95)^2 = 0.88$ (88%)
- Step $t+5$: weight = $(0.99 \times 0.95)^5 = 0.68$ (68%)
- Step $t+10$: weight = $(0.99 \times 0.95)^{10} = 0.47$ (47%)

**Parameters:**
- $\gamma = 0.99$: Discount factor for future rewards
- $\lambda = 0.95$: GAE parameter (bias-variance tradeoff)

**Implementation:**
- Class: [`RolloutBuffer`](rollout_buffer.py#L13) in [rollout_buffer.py](rollout_buffer.py)
- Computes advantages backward in time
- Normalizes advantages per batch for stability

**Mario Example:**

Consider a 3-step episode where Mario reaches the flag:

**Step-by-step breakdown:**

**Timestep t=0:** Mario moving forward
- State value: $V(s_0) = 100$
- Action: "right"
- Reward: $r_0 = 1$ (distance traveled)
- Next state value: $V(s_1) = 110$
- Done: $\text{done}_0 = 0$ (episode continues)
- **TD error:** $\delta_0 = 1 + 0.99 \times 110 \times (1-0) - 100 = \boxed{9.9}$

**Timestep t=1:** Mario still moving forward
- State value: $V(s_1) = 110$
- Action: "right"
- Reward: $r_1 = 1$ (distance traveled)
- Next state value: $V(s_2) = 120$
- Done: $\text{done}_1 = 0$ (episode continues)
- **TD error:** $\delta_1 = 1 + 0.99 \times 120 \times (1-0) - 110 = \boxed{9.8}$

**Timestep t=2:** Mario reaches the flag!
- State value: $V(s_2) = 120$
- Action: "right"
- Reward: $r_2 = 1000$ (flag bonus!)
- Next state value: N/A (episode ends)
- Done: $\text{done}_2 = 1$ (episode terminates)
- **TD error:** $\delta_2 = 1000 + 0.99 \times V(s_3) \times (1-1) - 120 = 1000 + 0 - 120 = \boxed{880}$

**Computing GAE advantages (backward recursion as implemented in code):**

The code computes advantages **backward in time** using the recursive formula:

$$\hat{A}_t = \delta_t + (\gamma\lambda) \cdot (1 - \text{done}_t) \cdot \hat{A}_{t+1}$$

This is more efficient than the forward formula (O(T) vs O(T²)) and naturally handles episode boundaries.
- **Forward**: For each timestep t, must sum all future TD errors from t to T
  - t=0: sum T terms, t=1: sum T-1 terms, ..., total = T + (T-1) + ... + 1 = T(T+1)/2 = **O(T²)**
- **Backward**: Single pass from T to 0, accumulating advantages
  - Each timestep: one addition operation, total = T operations = **O(T)**

**Backward computation (from t=2 to t=0):**

**Step 1: Process t=2 (last timestep, start here)**
- Initialize: $\hat{A}_3 = 0$ (no future advantage beyond episode end)
- Compute: $\hat{A}_2 = \delta_2 + (\gamma\lambda) \cdot (1 - \text{done}_2) \cdot \hat{A}_3$
- $\hat{A}_2 = 880 + (0.99 \times 0.95) \cdot (1-1) \cdot 0 = 880 + 0 = \boxed{880}$
- The $(1 - \text{done}_2) = 0$ term zeros out future advantages since episode ends

**Step 2: Process t=1 (working backward)**
- Compute: $\hat{A}_1 = \delta_1 + (\gamma\lambda) \cdot (1 - \text{done}_1) \cdot \hat{A}_2$
- $\hat{A}_1 = 9.8 + (0.99 \times 0.95) \cdot (1-0) \cdot 880$
- $\hat{A}_1 = 9.8 + 0.9405 \cdot 880 = 9.8 + 827.64 = \boxed{837.44}$

**Step 3: Process t=0 (final step, working backward)**
- Compute: $\hat{A}_0 = \delta_0 + (\gamma\lambda) \cdot (1 - \text{done}_0) \cdot \hat{A}_1$
- $\hat{A}_0 = 9.9 + (0.99 \times 0.95) \cdot (1-0) \cdot 837.44$
- $\hat{A}_0 = 9.9 + 0.9405 \cdot 837.44 = 9.9 + 787.56 = \boxed{797.46} \approx 800$

**Verification (equivalent forward formula):**

The result is identical to the forward formula:
$$\hat{A}_0 = \delta_0 + (\gamma\lambda) \cdot \delta_1 + (\gamma\lambda)^2 \cdot \delta_2 = 9.9 + 9.2 + 778.4 = 797.5 \approx 800$$

**Key insight:** Even though the immediate reward at $t=0$ was just +1, the advantage $\hat{A}_0 \approx 800$ is very high because GAE propagates the large positive surprise from reaching the flag ($\delta_2 = 880$) **backwards through time**. This tells the policy: "The action at $t=0$ was great because it eventually led to the flag!"

**Why backward computation?**
- **Efficiency**: O(T) single pass vs O(T²) for forward computation
- **Natural terminal handling**: $(1-\text{done}_t)$ automatically zeros out advantages beyond episode end
- **Code simplicity**: Single loop with accumulation variable

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

**Code Location:** [`PPOAgent.update()`](ppo_agent.py#L277-L285) in [ppo_agent.py](ppo_agent.py)

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

**Understanding the components:**

**1. What is $V_\phi(s_t)$?**
- $V_\phi(s_t)$ is the **current value prediction** from the critic network
- $\phi$ represents the critic's neural network parameters (weights)
- This is what the critic **currently thinks** the state is worth
- It's computed by forward-passing $s_t$ through the critic network
- During training, we update $\phi$ to make these predictions more accurate

**2. Why is $V^{target}_t = \hat{A}_t + V_{old}(s_t)$?**

This comes from the definition of advantage:

$$\hat{A}_t = \text{actual return} - V_{old}(s_t)$$

Rearranging:
$$\text{actual return} = \hat{A}_t + V_{old}(s_t) = V^{target}_t$$

**Intuitive explanation:**
- $V_{old}(s_t)$: What the **old critic** thought this state was worth (baseline)
- $\hat{A}_t$: How much **better or worse** things turned out compared to that baseline
- $\hat{A}_t + V_{old}(s_t)$: The **actual observed return** = baseline + surprise

**Example:**
- Old critic predicted: $V_{old}(s_t) = 100$
- Things went better than expected: $\hat{A}_t = +50$
- Target for new critic: $V^{target}_t = 100 + 50 = 150$
- The new critic should learn that this state is actually worth 150, not 100

**Why not use actual Monte Carlo returns directly?**
- Pure Monte Carlo returns have **high variance** (noisy)
- GAE advantages $\hat{A}_t$ are **lower variance** (smoothed through exponential weighting)
- Using $\hat{A}_t + V_{old}$ gives us a **bias-variance tradeoff**
- This is more stable for training than raw returns
- Better gradient estimates for policy updates

**Implementation:**
```python
# Compute value loss
value_pred = critic(states)
value_target = advantages + old_values  # Returns estimate
value_loss = F.mse_loss(value_pred, value_target.detach())
```

**Code Location:** [`PPOAgent.update()`](ppo_agent.py#L291) in [ppo_agent.py](ppo_agent.py)

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

**Code Location:** [`PPOAgent.update()`](ppo_agent.py#L296) in [ppo_agent.py](ppo_agent.py)

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

**Implementation:** [`PPOAgent.update()`](ppo_agent.py#L301-L305) in [ppo_agent.py](ppo_agent.py)

---

## 4. Training Process

This section describes the PPO training loop.

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
   - For K epochs:
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

**Code Location:** [`RolloutBuffer.get_batches()`](rollout_buffer.py#L320) in [rollout_buffer.py](rollout_buffer.py)

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

**Code Location:** [`PPOAgent.update()`](ppo_agent.py#L312) (actor) and [`PPOAgent.update()`](ppo_agent.py#L318) (critic) in [ppo_agent.py](ppo_agent.py)

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

**Code Location:** [`PPOAgent.update()`](ppo_agent.py#L325) in [ppo_agent.py](ppo_agent.py)

**Why it works:**
- Prevents policy from changing too rapidly
- Maintains on-policy assumption (data collected with $\pi_{old}$)
- Complements clipping for conservative updates

---

## References

1. **PPO Paper**: Schulman et al., 2017. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
2. **GAE Paper**: Schulman et al., 2016. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
3. **OpenAI Spinning Up**: [PPO Documentation](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
4. **CleanRL Implementation**: [ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)

---