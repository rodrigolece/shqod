from setuptools import setup, find_packages
setup(
    name='shqod',
    version='0.1',
    packages=find_packages(),
    # Do not run from zip file
    zip_safe=False,

    # Dependencies
    install_requires=['numpy>=1.17', 'scipy>=1.3', 'pandas>=1.0'],
    
    # Metadata
    author='Rodrigo L.C.',
    description='Origin-Destination tools to work with the SHQ dataset',
    license='MIT'
)
