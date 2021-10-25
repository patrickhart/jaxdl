"""Reinforcement learning replay buffer implementations"""
import gym
import collections
import numpy as np

from typing import Union


Batch = collections.namedtuple(
  'Batch',
  ['observations', 'actions', 'rewards', 'masks', 'next_observations'])

class ReplayBuffer:
  """ReplayBuffer for reinforcement learning agents."""
  def __init__(self,
    observation_space: gym.spaces.Box,
    action_space: Union[gym.spaces.Discrete, gym.spaces.Box],
    capacity: int):

    # initialize empty numpy arrays
    self.observations = np.empty(
      (capacity, *observation_space.shape), dtype=observation_space.dtype)
    self.actions = np.empty(
      (capacity, *action_space.shape), dtype=action_space.dtype)
    self.rewards = np.empty((capacity, ), dtype=np.float32)
    self.masks = np.empty((capacity, ), dtype=np.float32)
    self.terminals = np.empty((capacity, ), dtype=np.float32)
    self.next_observations = np.empty(
      (capacity, *observation_space.shape), dtype=observation_space.dtype)
    self.current_size = 0
    self.insertion_index = 0
    self.capacity = capacity

  def insert(
    self, observation: np.ndarray, action: np.ndarray,
    reward: float, mask: float, terminal: float,
    next_observation: np.ndarray):
    """Insert experience into replay buffer

    Args:
        observation (np.ndarray): Observed state of the evironment
        action (np.ndarray): Taken action
        reward (float): Reward obtained from the environment
        mask (float): Mask used to split the trajectories
        terminal (float): Whether the state is terminal
        next_observation (np.ndarray): Next observation state
    """
    # insert into buffer
    self.observations[self.insertion_index] = observation
    self.actions[self.insertion_index] = action
    self.rewards[self.insertion_index] = reward
    self.next_observations[self.insertion_index] = next_observation
    self.masks[self.insertion_index] = mask
    self.terminals[self.insertion_index] = terminal

    self.insertion_index = (self.insertion_index + 1) % self.capacity
    self.current_size = min(self.current_size + 1, self.capacity)

  def sample(self, batch_current_size: int) -> Batch:
    """Sample experiences from the replay buffer

    Args:
        batch_current_size (int): Size of the batch.

    Returns:
        Batch: Batch with experiences
    """
    # random sample experiences
    index = np.random.randint(self.current_size, size=batch_current_size)
    return Batch(
      observations=self.observations[index],
      actions=self.actions[index],
      rewards=self.rewards[index],
      masks=self.masks[index],
      next_observations=self.next_observations[index])