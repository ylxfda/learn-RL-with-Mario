# 🧠 Learn Reinforcement Learning with Mario

Welcome to **Learn Reinforcement Learning with Mario** — an educational repository that teaches you modern **reinforcement learning (RL)** through hands-on PyTorch implementations, using Super Mario as your guide.

**What you'll find here:**
- ✅ **PPO (Proximal Policy Optimization)** — learns from real experience
- ✅ **DreamerV3** — learns by building a world model and "dreaming"

**Who is this for?**
Anyone curious about how AI agents learn to play games. No RL background required — we'll start from the basics and build up to state-of-the-art algorithms.

---

## 🎯 Learning Philosophy

This README is structured around **questions**, not just answers. Each section asks:

> 🤔 **"What problem are we trying to solve?"**

By understanding the *why* behind each algorithm, you'll gain intuition that goes beyond memorizing formulas.

---

## 🎮 Why Mario?

Before diving into algorithms, let's establish *why* Mario is the perfect teacher for RL.

**Mario's world has all the core elements of reinforcement learning:**
- **States**: What Mario sees (enemies, blocks, pipes)
- **Actions**: What Mario can do (move, jump, run)
- **Rewards**: What Mario gets (coins, defeating enemies, reaching the flag)
- **Goal**: Mario must learn a strategy that maximizes long-term success

This is exactly what reinforcement learning is about.

---

## 🧩 What Is Reinforcement Learning?

**The Core Idea:**
Reinforcement learning teaches an agent to make decisions through trial and error, guided by rewards.

**The Loop:**
At each time step \( t \):

1. Agent observes **state** \( s_t \)
2. Agent selects **action** \( a_t \) based on its policy \( \pi_\theta(a|s) \)
3. Environment returns **reward** \( r_t \) and next **state** \( s_{t+1} \)
4. Agent updates its policy to get better rewards in the future

**The Objective:**
Learn a policy \( \pi_\theta \) that maximizes **expected cumulative reward**:

$$
J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right] \tag{1}
$$

where \( \gamma \in [0,1) \) is the discount factor (future rewards matter less than immediate ones).

---

## 📚 Learning About RL Algorithms

**Want to understand the evolution from simple policy gradients to world models?**

We've prepared a comprehensive, question-driven introduction that traces the journey from REINFORCE to PPO and DreamerV3. No prior RL knowledge required!

👉 **[Read: A Brief Introduction to Reinforcement Learning](A-Brief-Introduction-to-RL.md)**

This guide covers:
- REINFORCE and the basics of policy gradients
- Baselines and advantage functions
- Actor-Critic methods
- A2C/A3C parallel learning
- TRPO's trust region constraints
- PPO's clipped objective
- DreamerV3's world models and imagination
- Complete references and further reading

Each section asks **"What problem are we solving?"** and shows how each algorithm builds on previous insights.

---

## 🧰 Repository Structure

```
dreamerv3-torch-mario-claude/
│
├── PPO/
│   ├── ppo_agent.py          # PPO agent implementation
│   ├── networks.py            # Actor and Critic networks
│   ├── rollout_buffer.py     # Experience storage for PPO
│   └── README.md              # Detailed PPO documentation
│
├── dreamerv3/
│   ├── world_model.py         # RSSM world model implementation
│   ├── actor_critic.py        # Actor-Critic for DreamerV3
│   ├── networks/              # Neural network components
│   │   ├── rssm.py           # Recurrent State Space Model
│   │   └── encoder_decoder.py # CNN encoder/decoder
│   ├── utils/                 # Helper utilities
│   │   ├── distributions.py  # Probability distributions
│   │   └── tools.py          # Training utilities
│   └── README.md              # Detailed DreamerV3 documentation
│
├── envs/
│   ├── mario.py               # Mario environment wrapper
│   └── vec_mario.py           # Vectorized environments
│
├── configs/
│   ├── ppo_config.yaml        # PPO hyperparameters
│   └── dreamer_config.yaml    # DreamerV3 hyperparameters
│
├── train_mario_ppo.py         # Training script for PPO
├── train_mario_dreamer.py     # Training script for DreamerV3
├── play_mario_ppo.py          # Visualize trained PPO agent
├── play_mario_dreamer.py      # Visualize trained DreamerV3 agent
│
└── README.md                   # This file
```

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (recommended for faster training)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dreamerv3-torch-mario-claude.git
cd dreamerv3-torch-mario-claude

# Install dependencies
pip install -r requirements.txt
```

**Dependencies include:**
- `torch` — Deep learning framework
- `gymnasium` — RL environment interface
- `gym-super-mario-bros` — Super Mario environment
- `tensorboard` — Training visualization
- `numpy`, `PyYAML` — Utilities

---

## 🚀 Training Your First Agent

### Train PPO

```bash
python train_mario_ppo.py
```

**What happens:**
- Mario starts training in World 1-1
- Training progress is logged to `logdir/ppo/`
- Checkpoints are saved every 100 episodes
- Watch training curves in TensorBoard:
  ```bash
  tensorboard --logdir logdir/ppo
  ```

**Typical training time:** 2-4 hours on a modern GPU

---

### Train DreamerV3

```bash
python train_mario_dreamer.py
```

**What happens:**
- Collects initial experience
- Trains world model to predict observations
- Trains actor-critic in imagination
- Logs to `logdir/dreamer/`
- View world model reconstructions and training metrics in TensorBoard:
  ```bash
  tensorboard --logdir logdir/dreamer
  ```

**Typical training time:** 4-6 hours on a modern GPU

---

## 🎮 Playing with Trained Agents

Once training is complete, visualize what your agent learned:

```bash
# Watch PPO agent play
python play_mario_ppo.py --checkpoint logdir/ppo/checkpoints/best_model.pt

# Watch DreamerV3 agent play
python play_mario_dreamer.py --checkpoint logdir/dreamer/checkpoints/best_model.pt
```

---

## 📊 Understanding the Outputs

Both training scripts log:
- **Episode return**: Total reward per episode
- **Episode length**: How far Mario got
- **Success rate**: Percentage of episodes where Mario reached the flag
- **Loss curves**: Policy loss, value loss, world model loss (for DreamerV3)

**PPO-specific metrics:**
- Entropy (exploration level)
- KL divergence (policy change magnitude)
- Clip fraction (how often the clip is active)

**DreamerV3-specific metrics:**
- Reconstruction error (how well the model predicts observations)
- Reward prediction accuracy
- Imagination return vs. real return

---

## 🎓 Learning Path Recommendations

**For beginners:**
1. Read this README carefully
2. Start with PPO — it's simpler and more intuitive
3. Read [`PPO/README.md`](PPO/README.md) for implementation details
4. Run `train_mario_ppo.py` and watch Mario learn
5. Experiment with hyperparameters in `configs/ppo_config.yaml`

**After understanding PPO:**
1. Read about world models and model-based RL
2. Study the DreamerV3 architecture in [`dreamerv3/README.md`](dreamerv3/README.md)
3. Run `train_mario_dreamer.py`
4. Compare sample efficiency: how many frames does each algorithm need?

**Advanced explorations:**
- Implement your own environment wrappers
- Try different reward shaping strategies
- Experiment with different world model architectures
- Implement other algorithms (SAC, TD3, MuZero)

---

## 🔬 Key Implementation Details

### PPO Implementation Highlights

- **Generalized Advantage Estimation (GAE)**: Used for computing advantages with \( \lambda = 0.95 \)
- **Multiple epochs**: Reuses each batch of data for 4-10 gradient steps
- **Mini-batch updates**: Splits experience into smaller batches for stability
- **Entropy bonus**: Encourages exploration in early training

### DreamerV3 Implementation Highlights

- **RSSM (Recurrent State Space Model)**: Uses both deterministic and stochastic latent states
- **Free bits**: Prevents posterior collapse by ensuring KL divergence doesn't go too low
- **Symlog predictions**: Predicts rewards in symlog space for better numerical stability
- **\( \lambda \)-returns**: Uses exponential averaging for target values

---

## 🛠️ Troubleshooting

**Mario doesn't learn / reward stays near zero:**
- Check that the environment is rendering correctly
- Reduce learning rate
- Increase entropy coefficient (more exploration)
- Train for longer

**Training is very slow:**
- Enable CUDA if you have a GPU
- Reduce number of parallel environments
- Use smaller network architectures

**World model (DreamerV3) reconstructions look poor:**
- Increase world model training steps per environment step
- Increase KL divergence weight (balance reconstruction vs. latent regularization)
- Check that observations are properly normalized

**Out of memory errors:**
- Reduce batch size
- Reduce number of parallel environments
- Use gradient accumulation

---

## 💬 Contributing

Contributions are welcome! Whether it's:
- Bug fixes
- New features
- Documentation improvements
- Additional algorithms
- Better hyperparameters

Please open an issue or pull request.

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **OpenAI** for the PPO algorithm and Spinning Up resources
- **Danijar Hafner** for DreamerV3 and insightful papers
- **Gymnasium** and **gym-super-mario-bros** developers
- The RL community for open-source implementations and educational content

---

## 💭 Final Thoughts

> **PPO taught Mario to learn steadily from real experience.**
> **DreamerV3 taught Mario to think and plan inside his imagination.**

The journey from REINFORCE to DreamerV3 shows us that progress in AI comes from asking the right questions and systematically addressing limitations. Each algorithm builds on the insights of the previous one.

**Now it's your turn.**
Clone this repo, run the code, break things, fix them, and most importantly — *understand why these algorithms work the way they do.*

Happy learning, and may your Mario reach the flag! 🚩

---

**Questions? Issues? Ideas?**
Open an issue on GitHub or reach out to the community.
