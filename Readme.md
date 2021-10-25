# JAXDL: JAX (Flax) Deep Learning Library

Simple and clean JAX/Flax deep learning algorithm implementations:

* Soft-Actor-Critic ([arXiv:1812.05905](https://arxiv.org/abs/1812.05905))
* Transformer ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762); planned)
* Unified Graph Network Blocks ([arXiv:1806.01261](https://arxiv.org/abs/1806.01261); planned)

If you use JAXDL in your work, please cite this repository as follows:

```misc
@misc{jaxdl,
  author = {Hart, Patrick},
  month = {10},
  title = {{JAXDL: JAX Deep Learning Algorithm Implementations.}},
  url = {https://github.com/patrickhart/jaxdl},
  year = {2021}
}
```


## Results / Benchmark

### Continous Control From States
| HalfCheetah-v2 | Ant-v2 |
| --- | --- |
| ![HalfCheetah-v2](https://raw.githubusercontent.com/patrickhart/jaxdl/master/utils/learning_curves/HalfCheetah-v2.png) | ![Ant-v2](https://raw.githubusercontent.com/patrickhart/jaxdl/master/utils/learning_curves/Ant-v2.png) |
| Reacher-v2 | Humanoid-v2 |
| ![Reacher-v2](https://raw.githubusercontent.com/patrickhart/jaxdl/master/utils/learning_curves/Reacher-v2.png) | ![Humanoid-v2](https://raw.githubusercontent.com/patrickhart/jaxdl/master/utils/learning_curves/Humanoid-v2.png) |


## Installation

Install JAXDL using PyPi `pip install jaxdl`.

To use MuJoCo 2.1 you need to run `pip install git+https://github.com/nimrod-gileadi/mujoco-py` and place the binaries of MuJoCo in `~/.mujoco/mujoco210`.


## Examples / Getting Started

To get started have a look in the [examples folder](./examples).

To train a reinforcement learning agent run

```bash
python examples/run_rl.py \
  --mode=train \
  --env_name=Ant-v2 \
  --save_dir=./tmp/
```

To visualize the trained agent use

```bash
python examples/run_rl.py \
  --mode=visualize \
  --env_name=Ant-v2 \
  --save_dir=./tmp/
```


## Tensorboard

Monitor the training run using:

```bash
tensorboard --logdir=/save_dir/
```


## Contributing

Contributions are welcome!
This repository is meant to provide clean and simple implementations – please consider this when contributing.