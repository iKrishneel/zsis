#! /usr/bin/env python

from setuptools import setup
from setuptools import find_packages


try:
    with open('README.md', 'r') as f:
        readme = f.read()
except Exception:
    readme = str('')


install_requires = [
    'einops',
    'numpy',
    'matplotlib',
    'opencv-python',
    'pillow',
    'torch >= 1.8',
    'torchvision',
    'tqdm',
    'pytest',
    'clip  @ git+https://github.com/openai/CLIP.git@3702849800aa56e2223035bccd1c6ef91c704ca8'
]


setup(
    name='zsis',
    version='0.0.0',
    long_description=readme,
    packages=find_packages(),
    zip_safe=False,
    install_requires=install_requires,
    test_suite='tests',
)
