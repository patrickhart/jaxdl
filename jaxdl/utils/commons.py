from typing import Any, Dict, Sequence, Tuple

import flax
import jax
import flax.linen as nn
from flax.training import train_state, checkpoints

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
InfoDict = Dict[str, float]
Module = nn.Module
TrainState = train_state.TrainState


def create_train_state(model: Module, inputs: Sequence[Any],
  tx: Any = None) -> TrainState:
  """Creates a trian state for a nn.Module

  Args:
    model (nn.Module): Flax network module
    inputs (Sequence[Any]): key and inputs
    tx (Any, optional): Optimizer to update network. Defaults to None.

  Returns:
    TrainState: Train state object
  """
  params = model.init(*inputs)
  return train_state.TrainState.create(
    apply_fn=model.apply, params=params, tx=tx)

def restore_train_state(state: TrainState,
  workdir: str, prefix: str) -> Tuple[Any, InfoDict]:
  """Restores a trained state"""
  return checkpoints.restore_checkpoint(workdir, state, prefix=prefix)

def save_train_state(state: TrainState,
  workdir: str, prefix: str) -> None:
  """Saves a trained state"""
  if jax.process_index() == 0:
    step = int(state.step)
    checkpoints.save_checkpoint(
      workdir, state, step, keep=3, overwrite=True, prefix=prefix)