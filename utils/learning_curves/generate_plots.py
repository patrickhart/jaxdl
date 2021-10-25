
import os
import numpy as np
import matplotlib.pyplot as plt
from absl import app, flags
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.style.use('bmh')


FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment name.')
flags.DEFINE_string('algorithm', 'SAC', 'Agent name.')
flags.DEFINE_string('save_dir', './tmp/', 'Save directory.')
flags.DEFINE_integer('seed', 19, 'Random seed.')

def main(_):
  env_name = FLAGS.env_name
  log_dir = os.path.join(FLAGS.save_dir, FLAGS.env_name, str(FLAGS.seed), 'tb')
  event_acc = EventAccumulator(log_dir)
  event_acc.Reload()
  _, step_nums, vals = zip(*event_acc.Scalars('evaluation/average_returns'))
  step_nums = np.asarray(step_nums)
  vals = np.asarray([vals])
  _, ax = plt.subplots(1, 1)
  # TODO: average over mult. random seeds
  mean = np.mean(vals, axis=0)
  std = np.std(vals, axis=0)
  ax.plot(step_nums, mean, linewidth=1)
  ax.fill_between(step_nums, mean - std, mean + std, alpha=0.25)
  ax.set_xlabel('Environment Steps ($\\times 10^6%$)')
  ax.set_ylabel('Episode Return')
  ax.set_title(env_name + " / " + FLAGS.algorithm)
  plt.tight_layout()
  plt.savefig("./utils/learning_curves/" + env_name + ".png")
  plt.show()

if __name__ == '__main__':
  app.run(main)