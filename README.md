# 🧠 Learn Reinforcement Learning with Mario

<p align="center">
  <img src="assets/wbx06.png" alt="Project hero illustration" width="800"/>
  <br>
  <em>Project hero illustration</em>
</p>

Welcome to **Learn Reinforcement Learning with Mario** — an educational repository that teaches you modern **reinforcement learning (RL)** through hands-on PyTorch implementations, using Super Mario as your guide.

**What you'll find here:**
- 📖 **A brief RL introduction** — question-driven guide from REINFORCE to modern world models
- 🤖 **Two RL algorithms:**
  - **PPO (Proximal Policy Optimization)** — learns from real experience
  - **DreamerV3** — learns by building a world model and "dreaming"

**Who is this for?**
Anyone curious about how AI agents learn to play games. No RL background required — we'll start from the basics and build up to state-of-the-art algorithms.

---

## 🎯 Learning Philosophy

We believe that **curiosity is the best teacher**.

Perhaps you're here because you wondered: *"How do computers learn to play games?"* You've heard it's through something called **reinforcement learning**, but what exactly is that? And with so many RL algorithms out there—PPO and DreamerV3 were mentioned above—what makes them different?

👉 [Read: A Brief Introduction to Reinforcement Learning](A-Brief-Introduction-to-RL.md)

This question-driven guide walks you through the *why* behind each algorithm. You'll understand not just the formulas, but the **problems** each method solves and the **insights** that led to the next breakthrough. This guide is far from a comprehensive RL course, but it should be able to give you a quick overview of different RL algorithms and have you prepared to the next step.

Once you grasp the concepts, the next question probably is how theoretical ideas transform into working code? So far two RL algorithms have been implemented:
- **PPO** implementation in [`PPO/`](PPO/)
- **DreamerV3** implementation in [`dreamerv3/`](dreamerv3/)

As most variable and class naming follows the conventions from the original papers with detailed comments, the code is pretty much self-explanatory if you've read the papers.

---

## 🧰 Repository Structure

```
learn-rl-with-mario/
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



Once the enviroment is setup, you can start with either of the algorithms by reading the README file under the subdirectory.

Happy learning, and may your Mario reach the flag! 🚩

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.