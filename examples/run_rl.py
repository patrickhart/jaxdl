from typing import Any

import os
import gym
import random
import tqdm
from absl import app, flags
import numpy as np

from tensorboardX import SummaryWriter
from ml_collections import config_flags

from jaxdl.rl.agents.sac.sac import SACAgent
from jaxdl.rl.utils.replay_buffer import ReplayBuffer
from jaxdl.rl.environments.env_wrappers import make_env
from jaxdl.utils.commons import InfoDict
from jaxdl.rl.utils.commons import RLAgent


FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'Reacher-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/',
  'Save directory for Tensorboard logging and checkpoints.')
flags.DEFINE_integer('seed', 19, 'Environment random seed.')
flags.DEFINE_integer('eval_seed', 90, 'Evaluation environment random seed.')
flags.DEFINE_enum(
  name="mode", short_name="m", default="train",
  enum_values=["train", "visualize", "evaluate"], case_sensitive=False,
  help="Define mode to run.")
flags.DEFINE_integer(
  'eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Evaluation interval.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('training_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('initial_collection_episodes', int(1e4),
  'Number of initial off-policy collection steps before training.')
config_flags.DEFINE_config_file(
  'config',
  'examples/configs/sac_config.py',
  'Training parameters.',
  lock_config=False)

def setup_agent(env: gym.Env, config: config_flags):
  """Sets up an agent with the given environment and flags"""
  kwargs = dict(config)
  algorithm = kwargs.pop('algorithm')
  if algorithm == 'SAC':
    sac_agent = SACAgent(FLAGS.seed,
      env.observation_space.sample()[np.newaxis],
      env.action_space.sample()[np.newaxis], **kwargs)
    return sac_agent
  else:
    raise NotImplementedError()

def evaluate(agent: RLAgent, env: gym.Env, num_episodes: int,
  render: bool = False, temperature: float = 0.0) -> InfoDict:
  """Evaluates an agent"""
  evaluation_result = {'return': [], 'length': []}
  success_count = 0.
  info = {}
  for _ in range(num_episodes):
    observation, done = env.reset(), False
    while not done:
      action = agent.sample(observation, temperature=temperature)
      observation, _, done, info = env.step(action)
      if render:
        env.render()
    for key in evaluation_result.keys():
      evaluation_result[key].append(info['episode'][key])
    if 'is_success' in info:
      success_count += info['is_success']
  for key, value in evaluation_result.items():
    evaluation_result[key] = np.mean(value)
  evaluation_result['success'] = success_count / num_episodes
  return evaluation_result

def train(env: gym.Env, eval_env: gym.Env, replay_buffer_size: int,
  agent: Any, summary_writer: SummaryWriter, checkpoint_dir: str) -> None:
  """Trains an agent"""
  # replay buffer
  replay_buffer = ReplayBuffer(
    env.observation_space, env.action_space,
    replay_buffer_size or FLAGS.training_steps)

  observation, terminal = env.reset(), False
  for num_step in tqdm.tqdm(range(1, FLAGS.training_steps + 1), smoothing=0.1):
    # collect or sample
    if num_step < FLAGS.initial_collection_episodes:
      action = env.action_space.sample()
    else:
      action = agent.sample(observation)
    next_observation, reward, terminal, info = env.step(action)

    # fill replay buffer
    mask = 0.0
    if not terminal or 'TimeLimit.truncated' in info:
      mask = 1.0
    replay_buffer.insert(
      observation, action, reward, mask, float(terminal), next_observation)
    observation = next_observation

    if terminal:
      observation, terminal = env.reset(), False
      for key, value in info['episode'].items():
        summary_writer.add_scalar(f'training/{key}', value, info['timesteps'])
      if 'is_success' in info:
        summary_writer.add_scalar(
          f'training/success', info['is_success'], info['timesteps'])

    # train agent
    if num_step >= FLAGS.initial_collection_episodes:
      batch = replay_buffer.sample(FLAGS.batch_size)
      update_info = agent.update(batch)
      if num_step % FLAGS.log_interval == 0:
        for key, value in update_info.items():
          summary_writer.add_scalar(f'training/{key}', value, num_step)
        summary_writer.flush()

    # evaluate policy in eval_env
    if num_step % FLAGS.eval_interval == 0:
      evaluation_result = evaluate(agent, eval_env, FLAGS.eval_episodes)
      for key, value in evaluation_result.items():
        summary_writer.add_scalar(f'evaluation/average_{key}s',
          value, info['timesteps'])
      summary_writer.flush()
      agent.save(checkpoint_dir)


def main(_):
  np.random.seed(FLAGS.seed)
  random.seed(FLAGS.seed)

  # setup logging
  summary_writer = SummaryWriter(
    os.path.join(FLAGS.save_dir, FLAGS.env_name, str(FLAGS.seed), 'tb'))

  # make environments
  env = make_env(FLAGS.env_name, FLAGS.seed)
  eval_env = make_env(FLAGS.env_name, FLAGS.eval_seed)

  # setup agent
  config = dict(FLAGS.config)
  checkpoint_dir = os.path.join(
    FLAGS.save_dir, FLAGS.env_name, str(FLAGS.seed),
    'ckpts', config["algorithm"])
  replay_buffer_size = config.pop('replay_buffer_size')
  agent = setup_agent(env, config)
  agent.restore(checkpoint_dir)

  # switch between modes
  if FLAGS.mode == 'visualize':
    evaluate(agent, eval_env, FLAGS.eval_episodes, render=True)
  elif FLAGS.mode == 'evaluate':
    evaluate(agent, eval_env, FLAGS.eval_episodes, render=False)
  else:
    train(env, eval_env, replay_buffer_size, agent,
      summary_writer, checkpoint_dir)


if __name__ == '__main__':
  app.run(main)