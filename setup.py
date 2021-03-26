from setuptools import setup, find_packages
import sys

setup(name='chemistrygym',
      packages=find_packages(),
      install_requires=[
          'gym',
          'numpy',
          'matplotlib',
          'cmocean',
          'pyyaml',
          'pickle',
          'scipy'
      ],
      description='Implementation of extraction simulations in the OpenAI Gym environment framework.',
      author='CLEAN and UW ECE ML',
      url='https://github.com/chemgymrl/chemgymrl',
      version='0.0')
