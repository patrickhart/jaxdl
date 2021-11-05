import numpy as np
import unittest
from jaxdl.nlp.char_dataset import CharDataset


class TestNLPUtils(unittest.TestCase):
  def test_text_processing(self):
    text = "Hello there. How are you?! good, very good?"
    dataset = CharDataset(data=text, block_size=10)
    for _ in range(0, 10):
      idx = np.random.randint(0, len(dataset))
      label, target = dataset[idx]
      self.assertTrue(np.array_equal(label[1:], target[0:-1]))



if __name__ == '__main__':
  unittest.main()