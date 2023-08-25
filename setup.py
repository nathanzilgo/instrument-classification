from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='inda_mir',
    version='0.1',
    description='Indaband Music Information Retrieval',
    author='inda.band',
    author_email='nathan.pedroza@inda.band',
    packages=find_packages(),
    install_requires=install_requires,
)