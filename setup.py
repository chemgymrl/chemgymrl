from setuptools import setup, find_packages
import sys

setup(name='chemistrygym',
      packages=['chemistrylab'],
      install_requires=[
          'gym',
          'numpy',
      ],
      description='Implementation of extraction simulations in the OpenAI Gym environment framework.',
      author='CLEAN and UW ECE ML',
      url='https://github.com/CLEANit/chemistrygym',
      version='0.0')