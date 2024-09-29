from setuptools import setup, find_packages

setup(
    name='perlinrl',
    version='0.0.1',
    author='Anon',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'torch', 'perlin_noise', 'colorednoise', 'stable_baselines3'],
)
