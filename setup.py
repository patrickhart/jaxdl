import pathlib

from setuptools import find_packages, setup

folder_path = pathlib.Path(__file__).parent.resolve()
description = ('JAX (Flax) Deep Learning Library')
long_description = (folder_path / 'Readme.md').read_text(encoding='utf-8')

install_requires = [
    "tfp-nightly==0.15.0.dev20211020",
    "jax==0.2.24",
    "jaxlib==0.1.73",
    "flax==0.3.5",
    "chex==0.0.8",
    "pyyaml==6.0",
    "optax==0.0.9",
    "ml-collections==0.1.0",
    "gym==0.21.0",
    "tensorboardX==2.4.0",
    "tqdm==4.62.3",
    "tensorflow==2.7.0rc1",
    "numpy>=1.21.3"]

setup(
	name='jaxdl',
	version='0.0.2',
	description=description,
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='https://github.com/patrickahrt/jaxdl',
	author='Patrick Hart',
	classifiers=[
		'Development Status :: 3 - Alpha',
		'Intended Audience :: Developers',
		'Intended Audience :: Education',
		'Intended Audience :: Science/Research',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.6',
		'Programming Language :: Python :: 3.7',
		'Programming Language :: Python :: 3.8',
		'Programming Language :: Python :: 3.9',
		'Topic :: Scientific/Engineering :: Artificial Intelligence',
	],
	keywords='deep, machine, learning, reinforcement, research',
	packages=find_packages(),
	install_requires=install_requires,
	license='MIT',
  test_suite="tests"
)