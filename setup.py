from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='instrument-classification',
    version='0.1.0',
    description='TCC',
    url='',
    author='nathanzilgo',
    classifiers=[
        'Programming Language :: Python :: 3.11'
    ],
    install_requires=install_requires,
    python_requires='>=3.11'
)
