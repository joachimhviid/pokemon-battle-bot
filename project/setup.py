from setuptools import setup, find_packages

setup(
    name="pokemon_battle",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'numpy',
        'torch',
        'torchvision'
        'matplotlib',
    ],
)
