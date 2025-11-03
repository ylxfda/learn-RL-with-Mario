"""
PPO (Proximal Policy Optimization) Implementation for Super Mario Bros

This module implements the PPO algorithm from the paper:
"Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
https://arxiv.org/abs/1707.06347

PPO is an on-policy actor-critic algorithm that uses a clipped surrogate
objective to prevent destructively large policy updates. It has become one
of the most popular RL algorithms due to its simplicity, stability, and
strong empirical performance.

Key Components:
- networks.py: Actor (policy) and Critic (value) neural networks
- rollout_buffer.py: On-policy experience storage with GAE computation
- ppo_agent.py: Main PPO training algorithm

Main Differences from DreamerV3:
1. On-policy: PPO learns from recent experience only (no replay buffer)
2. Model-free: PPO learns policy directly without world model
3. Multiple parallel envs: PPO typically uses 8-128 parallel environments
4. Shorter rollouts: PPO uses 128-2048 step rollouts vs DreamerV3's longer sequences
"""

from PPO.networks import Actor, Critic
from PPO.rollout_buffer import RolloutBuffer
from PPO.ppo_agent import PPOAgent

__all__ = ['Actor', 'Critic', 'RolloutBuffer', 'PPOAgent']
