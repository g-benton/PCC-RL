from setuptools import setup
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))

if sys.version_info[0] < 3:
    with open(os.path.join(_here, 'README.rst')) as f:
        long_description = f.read()
else:
    with open(os.path.join(_here, 'README.rst'), encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='ntwrk',
    version='alpha',
    description=('Repo for Bayesian Optimization of Congestion Control'),
    long_description=long_description,
    author='Greg Benton, Nihar Sawant',
    author_email='',
    url='https://github.com/g-benton/demonstrating-congestion-control',
    license='Apache-2.0',
    packages=['ntwrk'],
   install_requires=[
    'setuptools==41.0.0',
    'torch>=1.0.1',
    'gpytorch>=0.3.1',
   ],
    include_package_data=True,
    classifiers=[
        'Development Status :: 0',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6'],
)
