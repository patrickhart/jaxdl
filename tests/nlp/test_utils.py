import gym
import numpy as np
from jax import random
import unittest
from jaxdl.nlp.utils import sequence_to_embeddings, embeddings_to_sequence, \
  sample_data

class TestNLPUtils(unittest.TestCase):
  def test_text_processing(self):
    text = "Hello there. How are you?! : good, better"
    encoded_text = sequence_to_embeddings(text)
    self.assertEqual(encoded_text.shape[0], len(text))
    decoded_text = embeddings_to_sequence(encoded_text)
    self.assertEqual(text, decoded_text)

  def test_data_generation(self):
    text = "Hello there. How are you?!"
    key = random.PRNGKey(0)
    source, target = sample_data(key, data=text, seq_len=10, batch_size=20)
    # print(embeddings_to_sequence(source[0]))
    # print(embeddings_to_sequence(target[0]))
    self.assertTrue((source[0, 1]==target[0, 0]).all())
    self.assertTrue(source.shape[0], target.shape[0])


if __name__ == '__main__':
  unittest.main()