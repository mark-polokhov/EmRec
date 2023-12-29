import os
from setuptools import find_packages, setup



if __name__ == '__main__':
    with open("version.txt") as f:
        version = f.read().strip()

    setup(
        name='Emotion Recognition',
        version='0.1',
        packages=find_packages(),
    )