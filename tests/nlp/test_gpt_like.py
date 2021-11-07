from jax._src.dtypes import dtype
import numpy as np
import unittest
import jax
import optax

from jaxdl.nlp.char_dataset import CharDataset
from jaxdl.nlp.gpt_like import GPTLike
from jaxdl.nn.transformers.transformer import Transformer, TransformerConfig
from jaxdl.nlp.transformer_fns import update_transformer
from jaxdl.utils.commons import create_train_state


class TestNLPUtils(unittest.TestCase):
  def test_text_processing(self):
    text = "Hello there. How are you?! good, very good?"
    dataset = CharDataset(data=text, block_size=10)

    transformer_config = TransformerConfig(
      vocab_size=dataset.vocab_size, block_size=dataset.block_size)
    transformer_net_fn = Transformer(transformer_config)
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)

    # init
    source = jax.random.uniform(
      key, shape=(10, dataset.block_size), dtype=np.float32)
    transformer_net = create_train_state(
      transformer_net_fn, [rng, source, key], optax.adam(learning_rate=0.001))
    rng, target = transformer_net.apply_fn(transformer_net.params, source, key)

    # test data stream
    source, target = dataset.get_sampled_batch(rng, 10)
    rng, new_transformer_net, info = update_transformer(
      transformer_net, source, target, key)


  def test_gpt_like_transformer(self):
    rng = jax.random.PRNGKey(0)
    text = "Hello there. How are you?! good, very good?"
    dataset = CharDataset(data=text, block_size=10)
    transformer_config = TransformerConfig(
      vocab_size=dataset.vocab_size, block_size=dataset.block_size)

    gpt = GPTLike(0, transformer_config)
    source, target = dataset.get_sampled_batch(rng, 10)

    # train step
    gpt.update(source, target)

    # TODO: sample
    # gpt.sample()




if __name__ == '__main__':
  unittest.main()