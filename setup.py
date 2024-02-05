import setuptools

setuptools.setup(
  name='utils',
  version='1.0',
  install_requires=[
    'apache-beam',
    'numpy',
    'pandas',
    'torch',
    'transformers',
  ],
  packages=setuptools.find_packages(),
)
