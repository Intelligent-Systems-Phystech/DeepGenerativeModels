import io
from setuptools import setup, find_packages

def read(file_path):
    with io.open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


__version__ = read('__version__')
readme = read('README.md')
requirements = read('requirements.txt')


setup(
    # metadata
    name='DeepGenerativeModels',
    version=__version__,
    author='Intelligent Systems',
    description='Lib of deep generative models',
    long_description=readme,
    
    # options
    packages=find_packages(),
    install_requires=requirements,
)