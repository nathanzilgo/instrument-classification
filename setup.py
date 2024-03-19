from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='inda_mir',
    version='0.2.0',
    description='A helper library for Music Information Retrieval tasks at indaband',
    url='https://github.com/indaband/track_classifier',
    author='inda.band',
    classifiers=[
        'Programming Language :: Python :: 3.11'
    ],
    packages=find_packages(include=['inda_mir']),
    install_requires=install_requires,
    python_requires='>=3.11'
)
