from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='src',
    version='0.1',
    description='Track Classifier',
    author='Nathan Fernandes Pedroza',
    author_email='nathan.pedroza@inda.band',
    packages=find_packages(),
    install_requires=install_requires,
)