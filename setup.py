"""
This file is part of ChemGymRL.

ChemGymRL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ChemGymRL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ChemGymRL.  If not, see <https://www.gnu.org/licenses/>.
"""
from setuptools import setup, find_packages
import sys

setup(name='chemistrygym',
      packages=find_packages(),
      install_requires=[
          'gymnasium',
          'numpy',
          'matplotlib>=3.6',
          'cmocean',
          'pyyaml',
          'scipy',
          'numba',
          'pillow',
          'pandas',
      ],
      include_package_data=True,
      description='Implementation of extraction simulations in the OpenAI Gym environment framework.',
      author='CLEAN and UW ECE ML',
      url='https://github.com/chemgymrl/chemgymrl',
      version='2.0.0')
